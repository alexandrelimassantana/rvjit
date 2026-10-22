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

#include "gtest/gtest.h"

#include "env.hpp"
#include "rvjit/implementation/rvv/linalg/matmul.hpp"
#include "rvjit/interface/linalg/matmul.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

using linalg::matmul::plan_t;

namespace {

constexpr int VLEN = 256;

plan_t extended_plan(type_t dta, type_t dtb, type_t dtb_reg, type_t dtc) {
    plan_t p;
    p.dta = dta;
    p.dtb = type_parameter_t(dtb, dtb_reg == type_t::none ? dtb : dtb_reg);
    p.dtc = dtc;
    p.m = size_parameter_t::less_eq(a3, 64);
    p.n = size_parameter_t::runtime(a4);
    p.k = size_parameter_t::runtime(a5);
    p.ptra = parameter_t(source_t(a0));
    p.ptrb = parameter_t(source_t(a1));
    p.ptrc = parameter_t(source_t(a2));
    p.lda = parameter_t(256);
    p.ldb = parameter_t(256);
    p.ldc = parameter_t(256);
    return p;
}

/// Whether a kernel for `p` can be planned
bool plans(const plan_t &p) {
    auto arch = rvv::buffer::architecture_t::default_list();
    arch.integer.without({a0, a1, a2, a3, a4, a5});
    rvv::vpu_t model;
    model.from_params(VLEN, 32768, 64, 4);
    host_model_env_t env(arch, model);
    rvv::linalg::matmul::impl_t impl(env.pm);
    linalg::matmul::matmul_t matmul(impl);
    return matmul.init(p);
}

} // namespace

/// B is in memory what `dtb.mem` says, and what `dtb.reg` says once loaded
TEST(matmul_b_extension, the_plan_names_both_types) {
    plan_t p
            = extended_plan(type_t::s32, type_t::s8, type_t::none, type_t::s32);
    EXPECT_TRUE(p.dtb.reg == type_t::s8);
    EXPECT_FALSE(p.dtb.converts());

    p.dtb.reg = type_t::s32;
    EXPECT_TRUE(p.dtb.widens());
    EXPECT_EQ(p.dtb.ratio(), 4);

    p.dtb.reg = type_t::s8;
    EXPECT_FALSE(p.dtb.converts());
}

/// A byte B multiplies into a uniform s32 or s16 accumulation, signed or not
TEST(matmul_b_extension, integer_extensions_plan) {
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::s8, type_t::s32, type_t::s32);
        EXPECT_TRUE(plans(p));
    }
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::u8, type_t::s32, type_t::s32);
        EXPECT_TRUE(plans(p));
    }
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::s16, type_t::s32, type_t::s32);
        EXPECT_TRUE(plans(p));
    }
    {
        const plan_t p = extended_plan(
                type_t::s16, type_t::s8, type_t::s16, type_t::s16);
        EXPECT_TRUE(plans(p));
    }
}

/// Without a compute type, the same pair is the macc shape rvjit has no form for
TEST(matmul_b_extension, a_narrow_b_without_a_compute_type_is_rejected) {
    const plan_t p
            = extended_plan(type_t::s32, type_t::s8, type_t::none, type_t::s32);
    EXPECT_FALSE(plans(p));
}

/// Only an integer extended to a wider integer, keeping its sign, is accepted
TEST(matmul_b_extension, other_extensions_are_rejected) {
    // A float B would need a conversion, not a free extension
    {
        const plan_t p = extended_plan(
                type_t::f32, type_t::f16, type_t::f32, type_t::f32);
        EXPECT_FALSE(plans(p));
    }
    // Not wider than the memory type
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::s16, type_t::s8, type_t::s32);
        EXPECT_FALSE(plans(p));
    }
    // A signed memory type needs a signed register type
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::s8, type_t::u16, type_t::s32);
        EXPECT_FALSE(plans(p));
    }
    // A register type the accumulation does not use
    {
        const plan_t p = extended_plan(
                type_t::s32, type_t::s8, type_t::s16, type_t::s32);
        EXPECT_FALSE(plans(p));
    }
}

} // namespace rvjit
