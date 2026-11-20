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
#include <utility>
#include <vector>

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

rvv::vpu_t machine_of(int vlen) {
    rvv::vpu_t model;
    model.from_params(vlen, 32768, 64, 4);
    return model;
}

rvv::vpu_t machine() {
    return machine_of(VLEN);
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

    {
        const auto p = f32_plan(size_parameter_t::literal(3),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, eighth));
    }
    EXPECT_TRUE(eighth.vtype.lmuli == LMUL::m1);

    // One element past the half group keeps the whole one
    {
        const auto p = f32_plan(size_parameter_t::literal(F32_VL / 2 + 1),
                size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
        ASSERT_TRUE(tune(p, over));
    }
    EXPECT_TRUE(over.vtype.lmuli == LMUL::m4);
    EXPECT_EQ(over.nb, 6);
}

/// The largest LMUL is the smallest group spanning 2048 bits, m8 below that
TEST_F(tuning_test_t, max_lmul_is_the_smallest_group_spanning_2048_bits) {
    const struct {
        int vlen;
        LMUL expected;
    } cases[] = {{128, LMUL::m8}, {256, LMUL::m8}, {512, LMUL::m4},
            {1024, LMUL::m2}, {2048, LMUL::m1}, {4096, LMUL::m1}};
    for (const auto &c : cases) {
        rvv::vpu_t model;
        model.from_params(c.vlen, 32768, 64, 4);
        EXPECT_TRUE(tuning_t(model).max_lmul() == c.expected)
                << "vlen=" << c.vlen;
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

namespace {

// A runtime M, N and K plan of `a` x `a` into `c`
plan_t typed_plan(const type_parameter_t &a, const type_parameter_t &c,
        const size_parameter_t &m = size_parameter_t::runtime(a3)) {
    plan_t p = f32_plan(
            m, size_parameter_t::runtime(a4), size_parameter_t::runtime(a5));
    p.dta = p.dtb = a;
    p.dtc = c;
    return p;
}

const type_parameter_t s8_in(type_t::s8, type_t::s32);
const type_parameter_t f16_out(type_t::f16, type_t::f32);

// Tunes `p` on a machine of `vlen` bits, returning the layout it grouped by
bool tune_at(int vlen, const plan_t &p, tuned_t &out,
        rvjit::buffer::vector_layout_t &layout) {
    rvv::buffer::architecture_t arch
            = rvv::buffer::architecture_t::default_list();
    rvv::buffer::buffer_pool_t pool(arch);
    const tuning_t tuning(machine_of(vlen));
    layout = tuning.vector_layout(p);
    if (!layout.is_valid() || !pool.configure_vectors(layout)) return false;
    return tuning(p, pool, {}, out);
}

} // namespace

/// At VLEN 1024 (m2 cap) a conversion from s8 takes half a register beside the
/// m2 group it widens into, instead of lifting the group to m4
TEST(tuning_lmul_cap, a_1024_bit_machine_keeps_s8_at_m2) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;

    ASSERT_TRUE(tune_at(1024, typed_plan(type_t::f32, type_t::f32), t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(t.m_grain, 64);
    EXPECT_EQ(layout.in, 2);

    // s8 data widens from s16 registers, e16/m1 into e32/m2; the s8 takes
    // half a register, a whole register beside each m1 input group
    ASSERT_TRUE(tune_at(1024, typed_plan(s8_in, type_t::s32), t, layout));
    EXPECT_TRUE(t.vtype.sewi == SEW::e16);
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m1);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(t.m_grain, 64);
    EXPECT_EQ(layout.in, 2);
    EXPECT_EQ(layout.out, 2);
}

/// Below 1024 bits the cap leaves room for whole groups: no conversion takes a
/// fraction of a register
TEST(tuning_lmul_cap, a_512_bit_machine_keeps_whole_groups) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    // s8 widens from e16/m2 into e32/m4, the s8 in the upper register
    ASSERT_TRUE(tune_at(512, typed_plan(s8_in, type_t::s32), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m2);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m4);
    EXPECT_EQ(layout.in, 2);
    EXPECT_EQ(layout.scratch, 0);
    EXPECT_EQ(t.nb, 7);

    ASSERT_TRUE(tune_at(512, typed_plan(type_t::f16, type_t::f32), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m2);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m4);
    EXPECT_EQ(t.nb, 7);
}

/// A short M narrows the group only while every conversion stays whole
TEST(tuning_lmul_cap, a_short_m_keeps_conversions_whole) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    const auto short_m = size_parameter_t::less_eq(a3, 32);
    ASSERT_TRUE(tune_at(
            1024, typed_plan(type_t::f32, type_t::f32, short_m), t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m1);

    // s8 at m1 would take a quarter register, where m2 takes half
    ASSERT_TRUE(
            tune_at(1024, typed_plan(s8_in, type_t::s32, short_m), t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
}

/// At VLEN 2048 (m1 cap) an f16 output takes half a register of the f32
/// group, and its C is widened into a scratch buffer of two registers
TEST(tuning_lmul_cap, a_2048_bit_machine_widens_an_f16_c_from_half_a_register) {
    plan_t p = typed_plan(type_t::f32, f16_out);
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    ASSERT_TRUE(tune_at(2048, p, t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m1);
    EXPECT_EQ(layout.scratch, 0);

    p.flags = linalg::matmul::with_beta;
    p.beta = scalar_parameter_t(1.0);
    ASSERT_TRUE(tune_at(2048, p, t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m1);
    EXPECT_EQ(layout.out, 1);
    EXPECT_EQ(layout.scratch, 2);
}

/// A narrow C or bias widened from a fraction of a register takes larger
/// scratch buffers, which the accumulators make room for
TEST(tuning_lmul_cap, a_fractional_epilogue_load_takes_larger_scratch) {
    plan_t p = typed_plan(s8_in, type_parameter_t(type_t::s8, type_t::s32));
    p.flags = linalg::matmul::with_beta;
    p.beta = scalar_parameter_t(1.0);
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    ASSERT_TRUE(tune_at(1024, p, t, layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(layout.scratch, 4);
}

/// No register group exceeds `max_lmul()`, and the model's choice
/// is never rejected for its LMUL
TEST(tuning_lmul_cap, no_group_exceeds_max_lmul) {
    const plan_t plans[] = {typed_plan(type_t::f32, type_t::f32),
            typed_plan(type_t::f16, type_t::f32),
            typed_plan(type_t::bf16, type_t::f32),
            typed_plan(type_t::f16, f16_out), typed_plan(s8_in, type_t::s32),
            typed_plan(s8_in, type_parameter_t(type_t::s8, type_t::s32))};
    for (int vlen : {128, 256, 512, 1024, 2048, 4096}) {
        const int cap = vgroup_size(tuning_t(machine_of(vlen)).max_lmul());
        for (const plan_t &p : plans) {
            tuned_t t;
            rvjit::buffer::vector_layout_t layout;
            ASSERT_TRUE(tune_at(vlen, p, t, layout)) << "vlen=" << vlen;
            EXPECT_LE(vgroup_size(t.vtype.lmuli), cap) << "vlen=" << vlen;
            EXPECT_LE(vgroup_size(t.vtype.lmulo), cap) << "vlen=" << vlen;
            EXPECT_EQ(lmul_fraction(t.vtype.lmulo), 1) << "vlen=" << vlen;
        }
    }
}

/// At VLEN 1024 (m2 cap) a widening takes m1 inputs into m2 accumulators
TEST(tuning_lmul_cap, a_1024_bit_machine_widens_f16_into_m2) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    ASSERT_TRUE(tune_at(1024, typed_plan(type_t::f16, type_t::f32), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m1);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(t.m_grain, 64);
}

/// At VLEN 2048 (m1 cap) a widening takes a fractional input LMUL, and s8
/// data a quarter register beside its s16 input
TEST(tuning_lmul_cap, a_2048_bit_machine_takes_fractions) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    ASSERT_TRUE(tune_at(2048, typed_plan(s8_in, type_t::s32), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::mf2);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m1);
    EXPECT_EQ(layout.in, 2);
    EXPECT_EQ(t.nb, 8);

    ASSERT_TRUE(tune_at(2048, typed_plan(type_t::f16, type_t::f32), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::mf2);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m1);
    EXPECT_EQ(t.m_grain, 2048 / 32);
    EXPECT_EQ(layout.in, 1);
}

/// A conversion that still takes a fraction of a register at `max_lmul` runs
/// there anyway
TEST(tuning_lmul_cap, a_conversion_that_cannot_be_whole_is_best_effort) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;
    // s8 data into s64 widens from s32 registers, e32/m1 into e64/m2: the s8
    // takes 256 bits at the m2 cap
    const type_parameter_t s8_wide(type_t::s8, type_t::s64);
    ASSERT_TRUE(tune_at(1024, typed_plan(s8_wide, type_t::s64), t, layout));
    EXPECT_TRUE(t.vtype.lmuli == LMUL::m1);
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m2);
    EXPECT_EQ(layout.in, 4); // a quarter of the m1 group in the fourth register
    EXPECT_EQ(t.nb, 12);
}

namespace {

// A plan of `a` x `b` into `c`, each of a memory and a register type
plan_t int_plan(type_t a_mem, type_t b_mem, type_t reg) {
    plan_t p = typed_plan(type_parameter_t(a_mem, reg), reg);
    p.dtb = type_parameter_t(b_mem, reg);
    return p;
}

} // namespace

/// Integer data at most half as wide as the accumulators is multiplied at
/// half their width, by a widening multiply-accumulate
TEST(tuning_widening, integer_data_widens_to_half_the_accumulator) {
    const tuning_t tuning(machine_of(256));

    const plan_t s8 = tuning.backend_plan(
            int_plan(type_t::s8, type_t::s8, type_t::s32));
    EXPECT_TRUE(s8.dta.reg == type_t::s16);
    EXPECT_TRUE(s8.dtb.reg == type_t::s16);
    EXPECT_TRUE(s8.dtc.reg == type_t::s32);
    EXPECT_TRUE(s8.dta.mem == type_t::s8);

    // Each operand keeps its memory type's signedness
    const plan_t u8 = tuning.backend_plan(
            int_plan(type_t::u8, type_t::u8, type_t::s32));
    EXPECT_TRUE(u8.dta.reg == type_t::u16);
    EXPECT_TRUE(u8.dtb.reg == type_t::u16);
    const plan_t mixed = tuning.backend_plan(
            int_plan(type_t::s8, type_t::u8, type_t::s32));
    EXPECT_TRUE(mixed.dta.reg == type_t::s16);
    EXPECT_TRUE(mixed.dtb.reg == type_t::u16);

    // s16 data takes no extension; s16 accumulators take s8 registers; s64
    // accumulators take s32 registers
    EXPECT_TRUE(
            tuning.backend_plan(int_plan(type_t::s16, type_t::s16, type_t::s32))
                    .dta.reg
            == type_t::s16);
    EXPECT_TRUE(
            tuning.backend_plan(int_plan(type_t::s8, type_t::s8, type_t::s16))
                    .dta.reg
            == type_t::s8);
    EXPECT_TRUE(
            tuning.backend_plan(int_plan(type_t::s8, type_t::s8, type_t::s64))
                    .dta.reg
            == type_t::s32);

    // The rule is idempotent
    const plan_t again = tuning.backend_plan(s8);
    EXPECT_TRUE(again.dta.reg == type_t::s16);
    EXPECT_TRUE(again.dtb.reg == type_t::s16);

    // Data as wide as half the accumulators or wider, floats, a widening plan
    // and a forced LMUL are run as they are
    EXPECT_TRUE(
            tuning.backend_plan(int_plan(type_t::s32, type_t::s32, type_t::s32))
                    .dta.reg
            == type_t::s32);
    EXPECT_TRUE(
            tuning.backend_plan(int_plan(type_t::s8, type_t::s16, type_t::s16))
                    .dtb.reg
            == type_t::s16);
    EXPECT_TRUE(tuning.backend_plan(typed_plan(type_parameter_t(type_t::f16,
                                                       type_t::f32),
                                            type_t::f32))
                        .dta.reg
            == type_t::f32);
    tuning_t forced(machine_of(256));
    forced.set_lmul(LMUL::m4);
    EXPECT_TRUE(
            forced.backend_plan(int_plan(type_t::s8, type_t::s8, type_t::s32))
                    .dta.reg
            == type_t::s32);
}

/// Widened s8 takes e16 inputs into e32 accumulators at every VLEN
TEST(tuning_widening, s8_blocks_at_each_vlen) {
    struct want_t {
        int vlen;
        LMUL lmuli, lmulo;
        int m_grain, nb;
    };
    // A zero nb is not checked: the N ceiling decides it
    const want_t wants[] = {{128, LMUL::m2, LMUL::m4, 16, 7},
            {256, LMUL::m2, LMUL::m4, 32, 7},
            {1024, LMUL::m1, LMUL::m2, 64, 0}};
    for (const want_t &w : wants) {
        for (type_t b : {type_t::s8, type_t::u8}) {
            tuned_t t;
            rvjit::buffer::vector_layout_t layout;
            ASSERT_TRUE(tune_at(
                    w.vlen, int_plan(type_t::s8, b, type_t::s32), t, layout))
                    << "vlen=" << w.vlen;
            EXPECT_TRUE(t.vtype.sewi == SEW::e16) << "vlen=" << w.vlen;
            EXPECT_TRUE(t.vtype.sewo == SEW::e32) << "vlen=" << w.vlen;
            EXPECT_TRUE(t.vtype.lmuli == w.lmuli) << "vlen=" << w.vlen;
            EXPECT_TRUE(t.vtype.lmulo == w.lmulo) << "vlen=" << w.vlen;
            EXPECT_EQ(t.m_grain, w.m_grain) << "vlen=" << w.vlen;
            if (w.nb) EXPECT_EQ(t.nb, w.nb) << "vlen=" << w.vlen;
        }
    }
}

/// The kernel at a long VLEN uses the blocks the tuning returned
TEST(tuning_agreement, a_long_vector_kernel_uses_the_tuned_blocks) {
    const type_parameter_t types[][2]
            = {{type_t::f32, type_t::f32}, {type_t::f16, type_t::f32},
                    {type_t::f16, f16_out}, {s8_in, type_t::s32},
                    {s8_in, type_parameter_t(type_t::s8, type_t::s32)}};
    // A runtime M unrolls as a large bound; 131 unrolls two tiles at 1024
    const size_parameter_t ms[] = {size_parameter_t::runtime(a3),
            size_parameter_t::literal(131), size_parameter_t::less_eq(a3, 64)};
    std::vector<plan_t> plans;
    for (const auto &t : types)
        for (const auto &m : ms)
            plans.push_back(typed_plan(t[0], t[1], m));
    // Widened integer data of each signedness
    for (const auto &ab : {std::make_pair(type_t::u8, type_t::u8),
                 std::make_pair(type_t::s8, type_t::u8),
                 std::make_pair(type_t::u8, type_t::s8)})
        plans.push_back(int_plan(ab.first, ab.second, type_t::s32));
    for (int vlen : {1024, 2048}) {
        for (size_t i = 0; i < plans.size(); ++i) {
            plan_t p = plans[i];
            const int all_flags[] = {0, linalg::matmul::with_beta};
            for (int flags : all_flags) {
                SCOPED_TRACE(testing::Message() << "vlen=" << vlen << " plan="
                                                << i << " flags=" << flags);
                p.flags = flags;
                p.beta = p.dta.reg == type_t::s32
                        ? scalar_parameter_t(1.0)
                        : scalar_parameter_t(source_t(a7));

                // Both sides see the same registers: the plan's runtime
                // values live in a3..a7
                auto kernel_arch = rvv::buffer::architecture_t::default_list();
                kernel_arch.integer.without({a0, a1, a2, a3, a4, a5, a6, a7});
                rvv::buffer::architecture_t arch = kernel_arch;
                rvv::buffer::buffer_pool_t pool(arch);
                const tuning_t tuning(machine_of(vlen));
                ASSERT_TRUE(pool.configure_vectors(tuning.vector_layout(p)));
                const auto epilogue = [&](int mb) {
                    linalg::matmul::needs_t needs;
                    EXPECT_TRUE(linalg::matmul::chain_needs(p, mb, 1, needs));
                    return needs;
                };
                tuned_t asked;
                ASSERT_TRUE(tuning(p, pool, epilogue, asked))
                        << "vlen=" << vlen << " flags=" << flags;

                host_model_env_t env(kernel_arch, machine_of(vlen));
                rvv::linalg::matmul::impl_t impl(env.pm);
                linalg::matmul::matmul_t matmul(impl);
                ASSERT_TRUE(matmul.init(p))
                        << "vlen=" << vlen << " flags=" << flags;
                EXPECT_EQ(impl.mb, asked.mb);
                EXPECT_EQ(impl.nb, asked.nb);
                EXPECT_EQ(impl.m_grain, asked.m_grain);
            }
        }
    }
}

namespace {

// `p` with B's stride, in bytes
plan_t with_ldb(plan_t p, int ldb) {
    p.ldb = parameter_t(ldb);
    return p;
}

// A plan of `a` into `c` whose M spans `m` elements at B's stride `ldb`
plan_t strided_plan(
        const type_parameter_t &a, const type_parameter_t &c, int m, int ldb) {
    return with_ldb(typed_plan(a, c, size_parameter_t::literal(m)), ldb);
}

} // namespace

/// Far apart rows of B, with M spanning two tiles, take two tiles beside three
/// accumulators, at the default group
TEST(tuning_strided_b, a_far_apart_b_takes_two_tiles_and_three_accumulators) {
    tuned_t t;
    rvjit::buffer::vector_layout_t layout;

    // 16 KiB: 4096 elements and a power-of-two factor
    ASSERT_TRUE(tune_at(256,
            strided_plan(type_t::f32, type_t::f32, 2 * F32_VL, 16384), t,
            layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m4);
    EXPECT_EQ(t.mb, 2);
    EXPECT_EQ(t.nb, 3);
    EXPECT_EQ(t.kb, 8);
    EXPECT_EQ(t.m_grain, F32_VL);

    // 17 KiB: no aliasing, 4352 elements
    ASSERT_TRUE(tune_at(256,
            strided_plan(type_t::f32, type_t::f32, 2 * F32_VL, 17408), t,
            layout));
    EXPECT_EQ(t.mb, 2);
    EXPECT_EQ(t.nb, 3);

    // Widening inputs keep their m2 into m4, and the integer ones their s16
    ASSERT_TRUE(tune_at(256,
            strided_plan(type_t::f16, type_t::f32, 2 * F32_VL, 4096), t,
            layout));
    EXPECT_TRUE(t.vtype.lmulo == LMUL::m4);
    EXPECT_EQ(t.mb, 2);
    EXPECT_LE(t.nb, 4);
    ASSERT_TRUE(tune_at(256, strided_plan(s8_in, type_t::s32, 2 * F32_VL, 4096),
            t, layout));
    EXPECT_EQ(t.mb, 2);
    EXPECT_LE(t.nb, 4);

    // A runtime M spans the tiles its upper bound does
    plan_t bounded
            = with_ldb(typed_plan(type_t::f32, type_t::f32,
                               size_parameter_t::less_eq(a3, 8 * F32_VL)),
                    16384);
    ASSERT_TRUE(tune_at(256, bounded, t, layout));
    EXPECT_EQ(t.mb, 2);
    EXPECT_EQ(t.nb, 3);
}

/// One tile of M beside the most accumulators is tuned for a stride that is not
/// far apart or unknown, a transposed B, one tile of M, an unbounded M, long
/// vectors and a pinned block
TEST(tuning_strided_b, one_tile_is_tuned_elsewhere) {
    tuned_t t, plain;
    rvjit::buffer::vector_layout_t layout;
    const plan_t m2 = strided_plan(type_t::f32, type_t::f32, 2 * F32_VL, 16384);

    // 4 KiB is 1024 elements of f32
    ASSERT_TRUE(tune_at(256,
            strided_plan(type_t::f32, type_t::f32, 2 * F32_VL, 4096), t,
            layout));
    EXPECT_GT(t.nb, 4);

    plan_t runtime = m2;
    runtime.ldb = parameter_t(source_t(a6));
    ASSERT_TRUE(tune_at(256, runtime, t, layout));
    EXPECT_GT(t.nb, 4);

    plan_t transposed = m2;
    transposed.is_transb = true;
    ASSERT_TRUE(tune_at(256, transposed, t, layout));
    EXPECT_NE(t.nb, 3);

    // Columns that fit one pass of four accumulators keep one tile
    plan_t few = m2;
    few.n = size_parameter_t::less_eq(a4, 4);
    ASSERT_TRUE(tune_at(256, few, t, layout));
    EXPECT_EQ(t.mb, 1);
    EXPECT_EQ(t.nb, 4);
    few.n = size_parameter_t::less_eq(a4, 5);
    ASSERT_TRUE(tune_at(256, few, t, layout));
    EXPECT_EQ(t.mb, 2);

    // A runtime M without a bound spans one tile
    plan_t unbounded = with_ldb(typed_plan(type_t::f32, type_t::f32), 16384);
    ASSERT_TRUE(tune_at(256, unbounded, t, layout));
    EXPECT_EQ(t.mb, 1);
    EXPECT_GT(t.nb, 4);

    // One tile of M leaves no room for a second
    ASSERT_TRUE(tune_at(256,
            strided_plan(type_t::f32, type_t::f32, F32_VL, 16384), t, layout));
    EXPECT_EQ(t.mb, 1);
    EXPECT_GT(t.nb, 4);

    // 2048-bit vectors
    ASSERT_TRUE(tune_at(2048, m2, t, layout));
    ASSERT_TRUE(tune_at(2048, with_ldb(m2, 256), plain, layout));
    EXPECT_EQ(t.mb, plain.mb);
    EXPECT_EQ(t.nb, plain.nb);

    // A pinned N block is honored as it is
    plan_t pinned = m2;
    pinned.blocks.nb = 5;
    ASSERT_TRUE(tune_at(256, pinned, t, layout));
    EXPECT_EQ(t.nb, 5);
}

/// Most plans keep one tile with the most accumulators, and the choice switches
/// to two tiles beside fewer only for far apart rows of B with M spanning two
/// tiles, whatever the data type
TEST(tuning_strided_b, the_choice_switches_with_the_stride_and_m) {
    const type_parameter_t types[][2] = {{type_t::f32, type_t::f32},
            {type_t::f16, type_t::f32}, {s8_in, type_t::s32}};
    const int strides[]
            = {16, 64, 128, 256, 512, 1024, 1536, 2047, 2048, 2560, 4096, 8192};
    const int ms[] = {F32_VL, 2 * F32_VL, 8 * F32_VL, 0};
    int one_tile = 0, two_tiles = 0;
    for (const auto &ty : types) {
        // The default: one tile beside the most accumulators
        tuned_t base;
        rvjit::buffer::vector_layout_t layout;
        ASSERT_TRUE(tune_at(256,
                with_ldb(typed_plan(ty[0], ty[1],
                                 size_parameter_t::literal(8 * F32_VL)),
                        256),
                base, layout));
        ASSERT_EQ(base.mb, 1);
        ASSERT_GE(base.nb, 5);

        for (int el : strides) {
            const int ldb = el * rvjit::sewb(ty[0].mem);
            const bool far_apart = el >= 2048;
            for (int m : ms) {
                SCOPED_TRACE(testing::Message()
                        << "elements=" << el << " m=" << m
                        << " mem=" << static_cast<int>(ty[0].mem));
                plan_t p = m ? strided_plan(ty[0], ty[1], m, ldb)
                             : with_ldb(typed_plan(ty[0], ty[1]), ldb);
                tuned_t t;
                ASSERT_TRUE(tune_at(256, p, t, layout));
                if (far_apart && m != F32_VL && m != 0) {
                    EXPECT_EQ(t.mb, 2);
                    EXPECT_LT(t.nb, base.nb);
                    EXPECT_GE(t.nb, 2);
                    ++two_tiles;
                } else {
                    EXPECT_EQ(t.mb, base.mb);
                    EXPECT_EQ(t.nb, base.nb);
                    ++one_tile;
                }
            }
        }
    }
    // Both choices occur, and the default is the common one
    EXPECT_GT(two_tiles, 0);
    EXPECT_GT(one_tile, 2 * two_tiles);
}

/// A tile that the kernel then generates for is the one the tuning returned
TEST(tuning_strided_b, the_two_tile_kernel_uses_the_tuned_blocks) {
    const plan_t p = strided_plan(type_t::f32, type_t::f32, 2 * F32_VL, 16384);
    rvv::buffer::architecture_t kernel_arch
            = rvv::buffer::architecture_t::default_list();
    rvv::buffer::buffer_pool_t pool(kernel_arch);
    const tuning_t tuning(machine());
    ASSERT_TRUE(pool.configure_vectors(tuning.vector_layout(p)));
    tuned_t asked;
    ASSERT_TRUE(tuning(p, pool, {}, asked));
    ASSERT_EQ(asked.mb, 2);

    host_model_env_t env(kernel_arch, machine());
    rvv::linalg::matmul::impl_t impl(env.pm);
    linalg::matmul::matmul_t matmul(impl);
    ASSERT_TRUE(matmul.init(p));
    EXPECT_EQ(impl.mb, asked.mb);
    EXPECT_EQ(impl.nb, asked.nb);
    EXPECT_EQ(impl.m_grain, asked.m_grain);
}

} // namespace rvjit
