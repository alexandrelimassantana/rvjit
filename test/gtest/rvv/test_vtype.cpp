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

#include <cstdint>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"

namespace rvjit {
namespace rvv {

using namespace Xbyak_riscv;

namespace {

/// Minimum supported VLEN
///
/// @note Every avl below stays within the vlmax this implies for its SEW/LMUL
///     pair, so vl equals the requested avl on any real VLEN (always >= this)
constexpr int VLENB = 16;

} // namespace

/// Expectations concerning `vtype` uniform precision initialization
TEST(vtype_config, uniform_fields) {
    vtype_t vt;

    const auto sew = SEW::e32;
    const auto lmul = LMUL::m2;
    vt.for_uniform(sew, lmul);

    EXPECT_EQ(vt.sewi, sew);
    EXPECT_EQ(vt.sewo, sew);
    EXPECT_EQ(vt.lmuli, lmul);
    EXPECT_EQ(vt.lmulo, lmul);
    EXPECT_TRUE(vt.is_uniform());
    EXPECT_FALSE(vt.is_mixed());
}

/// Expectations concerning `vtype` mixed precision initialization
TEST(vtype_config, mixed_fields) {
    vtype_t vt;

    const auto sewi = SEW::e16;
    const auto sewo = SEW::e32;
    const auto lmul = LMUL::m2;
    const auto expected_lmulo = LMUL::m4;
    vt.for_mixed(sewi, sewo, lmul);

    EXPECT_EQ(vt.sewi, sewi);
    EXPECT_EQ(vt.sewo, sewo);
    EXPECT_EQ(vt.lmuli, lmul);
    EXPECT_EQ(vt.lmulo, expected_lmulo);
    EXPECT_FALSE(vt.is_uniform());
    EXPECT_TRUE(vt.is_mixed());
}

/// Switching width keeps the output's vector length, scaling its LMUL
TEST(vtype_jit, keep_vl_scales_the_output_lmul) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        env.pm.vtype.for_uniform(SEW::e32, LMUL::m4);
        const VTA vt = env.pm.vtype.vt;
        const VMA vm = env.pm.vtype.vm;
        EXPECT_TRUE(env.pm.set_vtype_keep_vl(SEW::e16));
        gen.vsetvli(x0, x0, SEW::e16, LMUL::m2, vt, vm);
        EXPECT_TRUE(env.pm.set_vtype_keep_vl(SEW::e8));
        gen.vsetvli(x0, x0, SEW::e8, LMUL::m1, vt, vm);
        EXPECT_TRUE(env.pm.set_vtype_keep_vl(SEW::e64));
        gen.vsetvli(x0, x0, SEW::e64, LMUL::m8, vt, vm);
        EXPECT_TRUE(env.pm.set_vtype_keep_vl(SEW::e32));
        gen.vsetvli(x0, x0, SEW::e32, LMUL::m4, vt, vm);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, 4);
}

/// A width whose LMUL would be fractional emits nothing
TEST(vtype_jit, keep_vl_rejects_a_fractional_lmul) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        env.pm.vtype.for_uniform(SEW::e32, LMUL::m1);
        EXPECT_FALSE(env.pm.set_vtype_keep_vl(SEW::e16));
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_EQ(kernel.getSize(), sizeof(uint32_t));
}

/// set_vtype with a default (invalid) avl emits nothing
TEST(vtype_config, default_avl_is_invalid) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        env.pm.vtype.for_uniform(SEW::e8, LMUL::m1);
        env.pm.set_vtype(a0); // avl left as a default value
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_EQ(kernel.getSize(), sizeof(uint32_t)); // only ret was issued
}

namespace {

/// Probe that calls `set_vtype` and returns the granted vector length value
rv64_jit_probe_t::generate_fn_t vtype_explicit_probe(
        SEW sewi, SEW sewo, LMUL lmuli, LMUL lmulo, vtype_t::cfg_t kind) {
    return [=](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        env.pm.vtype.sewi = sewi;
        env.pm.vtype.sewo = sewo;
        env.pm.vtype.lmuli = lmuli;
        env.pm.vtype.lmulo = lmulo;
        const Reg avl = a0;
        const Reg out = a1;
        const Reg gvl = a2;

        env.pm.set_vtype(gvl, avl, kind);
        gen.sd(gvl, out, 0);
        gen.ret();
    };
}

} // namespace

/// Expectations on selected `set_vtype` SEW/LMUL values when using `input` kind
TEST(vtype_jit, set_input) {
    const auto lmul = LMUL::m1;
    const auto sewi = SEW::e8;
    const auto sewo = SEW::e32;
    const auto kind = vtype_t::cfg_t::input;
    const auto avl = 16;
    const auto expected_gvl = 16;

    rv64_jit_probe_t kernel(vtype_explicit_probe(sewi, sewo, lmul, lmul, kind));

    int64_t out = -1;
    ASSERT_TRUE(kernel.create_kernel());
    kernel(avl, &out);
    EXPECT_EQ(out, expected_gvl);
}

/// Expectations on selected `set_vtype` SEW/LMUL values using `output` kind
TEST(vtype_jit, set_output) {
    const auto lmul = LMUL::m1;
    const auto sewi = SEW::e8;
    const auto sewo = SEW::e16;
    const auto kind = vtype_t::cfg_t::output;
    const auto avl = 8;
    const auto expected_gvl = 8;

    rv64_jit_probe_t kernel(vtype_explicit_probe(sewi, sewo, lmul, lmul, kind));

    int64_t out = -1;
    ASSERT_TRUE(kernel.create_kernel());
    kernel(avl, &out);
    EXPECT_EQ(out, expected_gvl);
}

namespace {

/// Probe kernel materializing avl directly into a register before set_vtype
rv64_jit_probe_t::generate_fn_t vtype_default_probe() {
    return [](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        static constexpr int vl = 8;

        const Reg avl = a1;
        const Reg out = a0;
        const Reg gvl = a2;
        env.pm.vtype.for_uniform(SEW::e8, LMUL::m1);
        env.pm.vtype.avl = avl;

        gen.li(avl, vl);
        env.pm.set_vtype(gvl);
        gen.sd(gvl, out, 0);
        gen.ret();
    };
}

} // namespace

/// Expectations of using an avl register housed on `vtype`
TEST(vtype_jit, set_uses_avl_register) {
    const auto expected = 8;

    rv64_jit_probe_t kernel(vtype_default_probe());

    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    kernel(&out);
    EXPECT_EQ(out, expected);
}

namespace {

/// Probe checking the conditional emission in the `set_vtype_if_mixed` method
rv64_jit_probe_t::generate_fn_t vtype_set_if_mixed_probe(
        bool mixed, vtype_t::cfg_t kind) {
    return [mixed, kind](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        static constexpr int kSentinelAvl = 10;
        static constexpr int kMixedAvl = 4;

        const Reg out = a0;
        const Reg tmp = a1;
        const Reg sentinel_avl = a2;
        const Reg mixed_avl = a3;

        const auto lmul = LMUL::m1;
        const auto sewi = SEW::e8;
        env.pm.vtype.avl = mixed_avl;
        if (mixed) {
            env.pm.vtype.for_mixed(sewi, SEW::e16, lmul); // real 2x pair
        } else {
            env.pm.vtype.for_uniform(sewi, lmul);
        }

        // Establish a known vlen via a reference vsetvli unrelated to vtype
        gen.li(sentinel_avl, kSentinelAvl);
        gen.vsetvli(tmp, sentinel_avl, SEW::e64, LMUL::m8, VTA::ta, VMA::ma);

        gen.li(mixed_avl, kMixedAvl);
        env.pm.set_vtype_if_mixed(kind);

        gen.csrr(tmp, CSR::vl);
        gen.sd(tmp, out, 0);
        gen.ret();
    };
}

} // namespace

/// Expectations of `set_vtype_if_mixed` on mixed precision micro-kernels
TEST(vtype_jit, set_if_mixed_emits_on_mixed) {
    const auto kind = vtype_t::cfg_t::output;
    const auto use_mixed = true;

    rv64_jit_probe_t kernel(vtype_set_if_mixed_probe(use_mixed, kind));

    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    kernel(&out);
    EXPECT_EQ(out, 4);
}

/// Expectations of `set_vtype_if_mixed` on uniform precision micro-kernels
TEST(vtype_jit, set_if_mixed_noops_on_uniform) {
    const auto kind = vtype_t::cfg_t::output;
    const auto use_mixed = false;

    rv64_jit_probe_t kernel(vtype_set_if_mixed_probe(use_mixed, kind));

    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    kernel(&out);
    EXPECT_EQ(out, 10);
}

namespace {

/// Probe that sets the output vtype from `avl`, clobbers `avl`, switches to
/// the input vtype keeping the vector length and returns the length in effect
rv64_jit_probe_t::generate_fn_t vtype_keep_vl_probe() {
    return [](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);
        env.pm.vtype.for_mixed(SEW::e16, SEW::e32, LMUL::m2);
        const Reg avl = a0;
        const Reg out = a1;
        const Reg gvl = a2;

        env.pm.vtype.avl = avl;
        env.pm.set_vtype(vtype_t::cfg_t::output);
        gen.li(avl, 1000);
        env.pm.set_vtype_keep_vl_if_mixed(vtype_t::cfg_t::input);
        gen.csrr(gvl, CSR::vl);
        gen.sd(gvl, out, 0);
        gen.ret();
    };
}

} // namespace

/// Switching between input and output vtype keeps vl, freeing the avl register
TEST(vtype_jit, keep_vl_ignores_a_clobbered_avl) {
    rv64_jit_probe_t kernel(vtype_keep_vl_probe());

    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    kernel(5, &out);
    EXPECT_EQ(out, 5);
}

} // namespace rvv
} // namespace rvjit
