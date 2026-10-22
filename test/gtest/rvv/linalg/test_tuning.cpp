/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <functional>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rvjit/implementation/rvv/linalg/matmul.hpp"
#include "rvjit/implementation/rvv/linalg/tuning.hpp"
#include "rvjit/interface/linalg/matmul.hpp"
#include "rvjit/interface/linalg/matmul/postop/chain.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

using linalg::matmul::plan_t;
using rvv::linalg::matmul::tuned_t;
using rvv::linalg::matmul::tuning_t;

namespace {

// VLEN 256: one f32 vector group at m4 holds 32 elements
constexpr int VLEN = 256;
constexpr int F32_VL = VLEN / 32 * 4;

rvv::vpu_t machine() {
    rvv::vpu_t model;
    model.from_params(VLEN, 32768, 64, 4);
    return model;
}

plan_t f32_plan(const size_parameter_t &m, const size_parameter_t &n,
        const size_parameter_t &k) {
    plan_t p;
    p.m = m;
    p.n = n;
    p.k = k;
    p.ptra = parameter_t(source_t(a0));
    p.ptrb = parameter_t(source_t(a1));
    p.ptrc = parameter_t(source_t(a2));
    p.lda = parameter_t(F32_VL * 4);
    p.ldb = parameter_t(64 * 4);
    p.ldc = parameter_t(F32_VL * 4);
    return p;
}

} // namespace

/// The host side of the tuning: a pool over the default registers, no emitter
struct tuning_test_t : pool_test_t {
    tuning_t tuning {machine()};

    // Groups the pool as the plan asks, then tunes
    bool tune(const plan_t &p, tuned_t &out,
            const std::function<linalg::matmul::needs_t(int)> &epilogue = {}) {
        const auto layout = tuning.vector_layout(p);
        if (!layout.is_valid()) return false;
        if (!pool.configure_vectors(layout)) return false;
        return tuning(p, pool, epilogue, out);
    }
};

/// The K unrolls are fixed, and M follows what is known about it
TEST_F(tuning_test_t, defaults) {
    tuned_t t;
    {
        const auto p = f32_plan(size_parameter_t::runtime(a3),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, t));
    }
    EXPECT_EQ(t.m_grain, F32_VL);
    EXPECT_EQ(t.kb, 8);
    EXPECT_EQ(t.ktb, 4);
    EXPECT_EQ(t.mb, 1); // N takes every accumulator, leaving M none
    EXPECT_GT(t.nb, 1); // N takes the accumulators it can
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m4);
    EXPECT_TRUE(t.vtype.is_uniform());
}

/// A bound lets the dimension shrink its block below the default
TEST_F(tuning_test_t, bounds_shrink_the_blocks) {
    tuned_t t;
    {
        const auto p = f32_plan(size_parameter_t::literal(F32_VL),
                size_parameter_t::less_eq(a4, 2),
                size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, t));
    }
    EXPECT_EQ(t.nb, 2);
    EXPECT_EQ(t.mb, 1);

    tuned_t wide;
    {
        const auto p = f32_plan(size_parameter_t::literal(F32_VL * 2),
                size_parameter_t::literal(1), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, wide));
    }
    EXPECT_EQ(wide.nb, 1);
    EXPECT_EQ(wide.mb, 2); // two vectors of M fit beside one column
}

/// M takes at most four of the vectors N leaves, unless pinned
TEST_F(tuning_test_t, the_m_unroll_is_capped) {
    // At m1, eight vectors of M fit beside one column
    tuning.set_lmul(LMUL::m1);
    const int vl = F32_VL / 4;
    plan_t p = f32_plan(size_parameter_t::literal(vl * 8),
            size_parameter_t::literal(1), size_parameter_t::runtime(a5));
    tuned_t capped, pinned;
    ASSERT_TRUE(tune(p, capped));
    EXPECT_EQ(capped.nb, 1);
    EXPECT_EQ(capped.mb, 4);

    p.blocks.mb = 8;
    ASSERT_TRUE(tune(p, pinned));
    EXPECT_EQ(pinned.mb, 8);
}

/// A runtime M unrolls over the vectors N leaves, as a large bound does
TEST_F(tuning_test_t, a_runtime_m_unrolls_as_a_large_bound) {
    tuning.set_lmul(LMUL::m1);
    const auto narrow_n = [](const size_parameter_t &m) {
        return f32_plan(
                m, size_parameter_t::literal(1), size_parameter_t::runtime(a5));
    };
    tuned_t runtime, bounded;
    ASSERT_TRUE(tune(narrow_n(size_parameter_t::runtime(a3)), runtime));
    ASSERT_TRUE(tune(narrow_n(size_parameter_t::less_eq(a3, 4032)), bounded));
    EXPECT_EQ(runtime.mb, 4);
    EXPECT_EQ(runtime.mb, bounded.mb);
    EXPECT_EQ(runtime.nb, bounded.nb);
}

/// A bounded M below one vector takes the narrowest group that holds it
TEST_F(tuning_test_t, a_short_m_narrows_the_group) {
    tuned_t half, eighth, over;
    {
        const auto p = f32_plan(size_parameter_t::less_eq(a3, F32_VL / 2),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, half));
    }
    EXPECT_TRUE(half.vtype.lmuli == LMUL::m2);
    EXPECT_EQ(half.m_grain, F32_VL / 2);
    EXPECT_EQ(half.mb, 1);
    EXPECT_EQ(half.nb, 14); // sixteen m2 groups: fourteen accumulators

    {
        const auto p = f32_plan(size_parameter_t::literal(3),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, eighth));
    }
    EXPECT_TRUE(eighth.vtype.lmuli == LMUL::m1);
    EXPECT_EQ(eighth.nb, 16);

    // One element past the half group keeps the whole one
    {
        const auto p = f32_plan(size_parameter_t::literal(F32_VL / 2 + 1),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, over));
    }
    EXPECT_TRUE(over.vtype.lmuli == LMUL::m4);
    EXPECT_EQ(over.nb, 6);
}

/// A VLEN of 2048 bits or more caps the N block at 8 accumulators, below 16
TEST_F(tuning_test_t, long_vectors_cap_the_n_block) {
    const auto p = f32_plan(size_parameter_t::literal(3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    for (int vlen : {1024, 2048, 4096}) {
        rvv::vpu_t model;
        model.from_params(vlen, 32768, 64, 4);
        const tuning_t long_tuning(model);
        rvv::buffer::buffer_pool_t long_pool(arch);
        ASSERT_TRUE(long_pool.configure_vectors(long_tuning.vector_layout(p)));

        tuned_t t;
        ASSERT_TRUE(long_tuning(p, long_pool, {}, t)) << "vlen=" << vlen;
        EXPECT_TRUE(t.vtype.lmuli == LMUL::m1) << "vlen=" << vlen;
        EXPECT_EQ(t.nb, vlen >= 2048 ? 8 : 16) << "vlen=" << vlen;

        plan_t pinned = p;
        pinned.blocks.nb = 9;
        rvv::buffer::buffer_pool_t pinned_pool(arch);
        ASSERT_TRUE(
                pinned_pool.configure_vectors(long_tuning.vector_layout(p)));
        EXPECT_EQ(long_tuning(pinned, pinned_pool, {}, t), vlen < 2048)
                << "vlen=" << vlen;
    }
}

/// A widening plan narrows its input group and keeps the 2x accumulators
TEST_F(tuning_test_t, a_short_m_narrows_a_widening_group) {
    plan_t p = f32_plan(size_parameter_t::less_eq(a3, F32_VL / 2),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.dta = p.dtb = type_t::f16;
    tuned_t t;
    ASSERT_TRUE(tune(p, t));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m1);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(t.m_grain, F32_VL / 2);
}

/// A forced LMUL is kept for a short M
TEST_F(tuning_test_t, a_forced_lmul_ignores_a_short_m) {
    const auto p = f32_plan(size_parameter_t::literal(4),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    tuning.set_lmul(LMUL::m4);
    tuned_t t;
    ASSERT_TRUE(tune(p, t));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m4);
}

/// A pin is used as it is
TEST_F(tuning_test_t, pins_are_honored) {
    plan_t p = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.blocks.mb = 2;
    p.blocks.nb = 3;
    p.blocks.kb = 4;
    tuned_t t;
    ASSERT_TRUE(tune(p, t));
    EXPECT_EQ(t.mb, 2);
    EXPECT_EQ(t.nb, 3);
    EXPECT_EQ(t.kb, 4);
    EXPECT_EQ(t.ktb, 2); // halved while not below kb
}

/// A pin the machine cannot honor rejects the plan instead of shrinking
TEST_F(tuning_test_t, pins_beyond_the_machine_are_rejected) {
    const plan_t base = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    tuned_t t;

    plan_t k = base;
    k.blocks.kb = 9;
    EXPECT_FALSE(tune(k, t));

    plan_t n = base;
    n.blocks.nb = 17;
    EXPECT_FALSE(tune(n, t));

    // Accumulators beside a wide N block leave no room for a second M vector
    plan_t m = base;
    m.blocks.nb = 8;
    m.blocks.mb = 4;
    EXPECT_FALSE(tune(m, t));
}

namespace {
linalg::matmul::needs_t epilogue_needs(
        int accs, int wanted = 0, bool mask = false) {
    linalg::matmul::needs_t n;
    n.accs = accs;
    n.accs_wanted = std::max(accs, wanted);
    n.mask = mask;
    return n;
}
} // namespace

/// The accumulators keep the pipeline groups; the epilogue fits the input ones
TEST_F(tuning_test_t, the_epilogue_reuses_the_input_registers) {
    const plan_t p = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    tuned_t bare, ringed, held;
    ASSERT_TRUE(tune(p, bare));
    // Eight m4 groups: six accumulators, two inputs
    EXPECT_EQ(bare.nb, 6);

    // A ring it would like to grow does not shrink the accumulators
    ASSERT_TRUE(tune(p, ringed, [](int) { return epilogue_needs(1, 4); }));
    EXPECT_EQ(ringed.nb, 6);

    // Both input groups, v0's included, are free to the epilogue
    ASSERT_TRUE(tune(p, held, [](int) { return epilogue_needs(2); }));
    EXPECT_EQ(held.nb, 6);
}

/// What the epilogue cannot do without leaves fewer groups to the accumulators
TEST_F(tuning_test_t, the_epilogue_minimum_shrinks_the_accumulators) {
    const plan_t p = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    tuned_t loaded, masked;
    ASSERT_TRUE(tune(p, loaded, [](int) { return epilogue_needs(3); }));
    EXPECT_EQ(loaded.nb, 5);

    // The mask keeps v0's group out of the epilogue's reach
    {
        const auto needs = [](int) { return epilogue_needs(3, 0, true); };
        ASSERT_TRUE(tune(p, masked, needs));
    }
    EXPECT_EQ(masked.nb, 4);

    // An epilogue that leaves nothing for the accumulators is rejected
    tuned_t none;
    EXPECT_FALSE(tune(p, none, [](int) { return epilogue_needs(64); }));
}

/// A widening kernel keeps seven accumulators and two m2 inputs
TEST_F(tuning_test_t, a_widening_epilogue_reuses_the_input_registers) {
    plan_t p = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.dta = p.dtb = type_t::f16;
    tuned_t bare, loaded;
    ASSERT_TRUE(tune(p, bare));
    EXPECT_EQ(bare.nb, 7);
    EXPECT_EQ(bare.vtype.lmuli, LMUL::m2);
    ASSERT_TRUE(tune(p, loaded, [](int) { return epilogue_needs(1, 2); }));
    EXPECT_EQ(loaded.nb, 7);
}

/// The built-in beta and bias epilogue plans in the input registers
TEST_F(tuning_test_t, the_builtin_epilogue_fits_beside_the_full_accumulators) {
    for (const type_t dta : {type_t::f32, type_t::f16}) {
        plan_t p = f32_plan(size_parameter_t::runtime(a3),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        p.dta = p.dtb = dta;
        p.flags = linalg::matmul::with_beta | linalg::matmul::with_bias;
        p.bias = parameter_t(source_t(a6));
        p.beta = scalar_parameter_t(source_t(a7));

        const auto epilogue = [&](int mb) {
            linalg::matmul::needs_t needs;
            EXPECT_TRUE(linalg::matmul::chain_needs(p, mb, 1, needs));
            return needs;
        };
        tuned_t t;
        ASSERT_TRUE(tune(p, t, epilogue));
        // m4 uniform: six accumulators, two inputs. m2 widening to m4: seven
        // accumulators, two inputs
        EXPECT_EQ(t.nb, dta == type_t::f32 ? 6 : 7);
        EXPECT_EQ(t.mb, 1);

        auto arch = rvv::buffer::architecture_t::default_list();
        arch.integer.without({a0, a1, a2, a3, a4, a5, a6, a7});
        host_model_env_t env(arch, machine());
        rvv::linalg::matmul::impl_t impl(env.pm);
        linalg::matmul::matmul_t matmul(impl);
        ASSERT_TRUE(matmul.init(p));
        EXPECT_EQ(impl.nb, t.nb);
    }
}

/// The blocks a tuning returns, pinned, are what the tuning returns again
TEST_F(tuning_test_t, tuned_blocks_are_a_fixed_point) {
    const size_parameter_t sizes[] = {size_parameter_t::runtime(a3),
            size_parameter_t::less_eq(a3, F32_VL * 3),
            size_parameter_t::literal(F32_VL * 2 + 5),
            size_parameter_t::literal(1)};
    for (const auto &m : sizes) {
        for (const auto &n : sizes) {
            plan_t p = f32_plan(m, n, size_parameter_t::runtime(a5));
            tuned_t first;
            ASSERT_TRUE(tune(p, first));

            p.blocks.mb = first.mb;
            p.blocks.nb = first.nb;
            p.blocks.kb = first.kb;
            tuned_t again;
            ASSERT_TRUE(tune(p, again));
            EXPECT_EQ(again.mb, first.mb);
            EXPECT_EQ(again.nb, first.nb);
            EXPECT_EQ(again.kb, first.kb);
            EXPECT_EQ(again.ktb, first.ktb);
        }
    }
}

/// A widening plan groups accumulators wider, and a too-large forced LMUL fails
TEST_F(tuning_test_t, widening_layout) {
    plan_t p = f32_plan(size_parameter_t::runtime(a3),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.dta = p.dtb = type_t::f16;
    p.dtc = type_t::f32;

    const auto layout = tuning.vector_layout(p);
    ASSERT_TRUE(layout.is_valid());
    EXPECT_EQ(layout.out, 2 * layout.in);

    tuning.set_lmul(LMUL::m8);
    EXPECT_FALSE(tuning.vector_layout(p).is_valid());
    tuned_t t;
    EXPECT_FALSE(tune(p, t));
}

/// A narrowed output keeps the vtype and the blocks of the wide one
TEST_F(tuning_test_t, a_narrowed_output_keeps_the_blocks) {
    plan_t wide = f32_plan(size_parameter_t::less_eq(a3, F32_VL * 2),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    wide.dta = wide.dtb = type_t::f16;
    wide.dtc = type_t::f32;
    tuned_t base;
    ASSERT_TRUE(tune(wide, base));

    plan_t narrow = wide;
    narrow.dtc = type_parameter_t(type_t::f16, type_t::f32);
    tuned_t t;
    ASSERT_TRUE(tune(narrow, t));
    EXPECT_TRUE(t.vtype.lmulo == base.vtype.lmulo);
    EXPECT_EQ(t.mb, base.mb);
    EXPECT_EQ(t.nb, base.nb);
}

/// An output converts only at widths that keep the accumulators' vector
/// length in a whole register group
TEST_F(tuning_test_t, an_output_must_fit_a_register_group) {
    plan_t p = f32_plan(size_parameter_t::literal(F32_VL),
            size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.dtc = type_parameter_t(type_t::s8, type_t::f32);
    tuned_t t;
    tuning.set_lmul(LMUL::m4);
    EXPECT_TRUE(tune(p, t));

    // At m2 an s8 output would need a fractional LMUL
    tuning.set_lmul(LMUL::m2);
    EXPECT_FALSE(tune(p, t));

    // A uniform f32 accumulation at m1 would narrow to a fractional LMUL
    tuning.set_lmul(LMUL::m1);
    p.dtc.mem = type_t::f16;
    EXPECT_FALSE(tune(p, t));
}

/// The blocks the kernel plan uses are the ones a host asked for beforehand
TEST(tuning_agreement, a_kernel_uses_the_blocks_the_tuning_returned) {
    const size_parameter_t ms[] = {size_parameter_t::less_eq(a3, F32_VL / 2),
            size_parameter_t::literal(3), size_parameter_t::less_eq(a3, F32_VL),
            size_parameter_t::less_eq(a3, F32_VL * 2),
            size_parameter_t::literal(F32_VL * 3)};
    for (const auto &m : ms) {
        const int all_flags[] = {0, linalg::matmul::with_bias,
                linalg::matmul::with_beta | linalg::matmul::with_bias};
        for (int flags : all_flags) {
            plan_t p = f32_plan(m, size_parameter_t::runtime(a4),
                    size_parameter_t::runtime(a5));
            p.flags = flags;
            p.bias = parameter_t(source_t(a6));
            p.beta = scalar_parameter_t(source_t(a7));

            // The host side, with no emitter
            rvv::buffer::architecture_t arch
                    = rvv::buffer::architecture_t::default_list();
            rvv::buffer::buffer_pool_t pool(arch);
            tuning_t tuning(machine());
            ASSERT_TRUE(pool.configure_vectors(tuning.vector_layout(p)));
            const auto epilogue = [&](int mb) {
                linalg::matmul::needs_t needs;
                EXPECT_TRUE(linalg::matmul::chain_needs(p, mb, 1, needs));
                return needs;
            };
            tuned_t asked;
            ASSERT_TRUE(tuning(p, pool, epilogue, asked));

            // The kernel side, deciding on its own
            auto kernel_arch = rvv::buffer::architecture_t::default_list();
            kernel_arch.integer.without({a0, a1, a2, a3, a4, a5, a6, a7});
            host_model_env_t env(kernel_arch, machine());
            rvv::linalg::matmul::impl_t impl(env.pm);
            linalg::matmul::matmul_t matmul(impl);
            ASSERT_TRUE(matmul.init(p));
            EXPECT_EQ(impl.mb, asked.mb);
            EXPECT_EQ(impl.nb, asked.nb);
            EXPECT_EQ(impl.kb, asked.kb);
            EXPECT_EQ(impl.ktb, asked.ktb);
            EXPECT_EQ(impl.m_grain, asked.m_grain);
        }
    }
}

} // namespace rvjit
