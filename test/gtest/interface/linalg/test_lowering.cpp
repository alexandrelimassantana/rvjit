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

#include <functional>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rvjit/implementation/rvv/linalg/matmul.hpp"
#include "rvjit/interface/linalg/matmul.hpp"
#include "rvjit/interface/linalg/matmul/lowering.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

using linalg::matmul::lowering_error;
using linalg::matmul::plan_t;
using linalg::matmul::preserves_values;

namespace {

// An s8 x s8 into s32 plan, registers holding s32
plan_t s8_plan() {
    plan_t p;
    p.dta = p.dtb = type_parameter_t(type_t::s8, type_t::s32);
    p.dtc = type_t::s32;
    p.m = size_parameter_t::literal(64);
    p.n = size_parameter_t::literal(4);
    p.k = size_parameter_t::literal(8);
    p.ptra = parameter_t(source_t(a0));
    p.ptrb = parameter_t(source_t(a1));
    p.ptrc = parameter_t(source_t(a2));
    p.lda = parameter_t(64);
    p.ldb = parameter_t(32);
    p.ldc = parameter_t(256);
    return p;
}

// The plan with A and B held in s16
plan_t s16_registers(plan_t p) {
    p.dta.reg = p.dtb.reg = type_t::s16;
    return p;
}

} // namespace

/// A register type holds the values of a narrower type of its kind: integers
/// of the same signedness, unsigned ones in a wider signed type, and floats
TEST(matmul_lowering, preserves_values) {
    EXPECT_TRUE(preserves_values(type_t::s8, type_t::s8));
    EXPECT_TRUE(preserves_values(type_t::s8, type_t::s16));
    EXPECT_TRUE(preserves_values(type_t::s8, type_t::s64));
    EXPECT_TRUE(preserves_values(type_t::u8, type_t::u16));
    EXPECT_TRUE(preserves_values(type_t::u8, type_t::s16));
    EXPECT_TRUE(preserves_values(type_t::u16, type_t::s32));
    EXPECT_TRUE(preserves_values(type_t::f16, type_t::f32));
    EXPECT_TRUE(preserves_values(type_t::bf16, type_t::f32));
    EXPECT_TRUE(preserves_values(type_t::f32, type_t::f64));

    EXPECT_FALSE(preserves_values(type_t::s8, type_t::u16));
    EXPECT_FALSE(preserves_values(type_t::u8, type_t::s8));
    EXPECT_FALSE(preserves_values(type_t::s32, type_t::s16));
    EXPECT_FALSE(preserves_values(type_t::s32, type_t::f32));
    EXPECT_FALSE(preserves_values(type_t::f32, type_t::s32));
    EXPECT_FALSE(preserves_values(type_t::f16, type_t::bf16));
    EXPECT_FALSE(preserves_values(type_t::f32, type_t::f16));
    EXPECT_FALSE(preserves_values(type_t::none, type_t::s8));
}

/// A plan lowers to itself, and to one holding A and B in a narrower type
TEST(matmul_lowering, accepts_register_type_changes_only) {
    const plan_t p = s8_plan();
    EXPECT_EQ(lowering_error(p, p), nullptr);
    EXPECT_EQ(lowering_error(p, s16_registers(p)), nullptr);

    plan_t unsigned_a = p;
    unsigned_a.dta = type_parameter_t(type_t::u8, type_t::s32);
    plan_t lowered = unsigned_a;
    lowered.dta.reg = type_t::u16;
    EXPECT_EQ(lowering_error(unsigned_a, lowered), nullptr);
}

/// Anything else that changes is rejected with its reason
TEST(matmul_lowering, rejects_every_other_change) {
    const plan_t p = s8_plan();
    const auto rejects = [&](const std::function<void(plan_t &)> &change) {
        plan_t q = s16_registers(p);
        change(q);
        return lowering_error(p, q) != nullptr;
    };

    EXPECT_TRUE(rejects([](plan_t &q) { q.dtc = type_t::s64; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dtc.mem = type_t::s16; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dta.mem = type_t::s16; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dtb.mem = type_t::u8; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dtepi = type_t::f32; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dtbias = type_t::s8; }));
    EXPECT_TRUE(rejects([](plan_t &q) {
        q.bias_kind = linalg::matmul::bias_kind_t::scalar;
    }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.is_transa = true; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.is_transb = true; }));
    EXPECT_TRUE(
            rejects([](plan_t &q) { q.flags = linalg::matmul::with_beta; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.blocks.nb = 2; }));

    // A register type that does not hold the memory type's values
    EXPECT_TRUE(rejects([](plan_t &q) { q.dta.reg = type_t::u16; }));
    EXPECT_TRUE(rejects([](plan_t &q) { q.dtb.reg = type_t::f32; }));
    plan_t narrowed = p;
    narrowed.dta = type_t::s32;
    narrowed.dta.reg = type_t::s16;
    EXPECT_NE(lowering_error(narrowed, narrowed), nullptr);
}

namespace {

// The RVV implementation lowering as a test says
struct lowering_impl_t : rvv::linalg::matmul::impl_t {
    explicit lowering_impl_t(rvv::programming_model_t &pm)
        : rvv::linalg::matmul::impl_t(pm) {}

    std::function<plan_t(const plan_t &)> lower;

    plan_t backend_plan(const plan_t &p) const override {
        return lower ? lower(p) : rvv::linalg::matmul::impl_t::backend_plan(p);
    }
};

// Whether `matmul_t::init` accepts `p` under an implementation lowering by
// `lower`
bool inits(
        const plan_t &p, const std::function<plan_t(const plan_t &)> &lower) {
    auto arch = rvv::buffer::architecture_t::default_list();
    arch.integer.without({a0, a1, a2, a3, a4, a5, a6, a7});
    rvv::vpu_t model;
    model.from_params(256, 32768, 64, 4);
    host_model_env_t env(arch, model);
    lowering_impl_t impl(env.pm);
    impl.lower = lower;
    linalg::matmul::matmul_t matmul(impl);
    return matmul.init(p);
}

} // namespace

/// `matmul_t::init` runs a lowering, and rejects what is not one or does not
/// hold when lowered again
TEST(matmul_lowering, init_enforces_the_contract) {
    const plan_t p = s8_plan();
    EXPECT_TRUE(inits(p, {}));
    EXPECT_TRUE(inits(p, s16_registers));

    // Not a lowering: the accumulators change
    EXPECT_FALSE(inits(p, [](const plan_t &q) {
        plan_t r = s16_registers(q);
        r.dtc = type_t::s64;
        return r;
    }));

    // A register type that does not hold the memory type
    EXPECT_FALSE(inits(p, [](const plan_t &q) {
        plan_t r = q;
        r.dta.reg = type_t::u16;
        return r;
    }));

    // Not stable: it narrows its own result once more
    EXPECT_FALSE(inits(p, [](const plan_t &q) {
        plan_t r = q;
        r.dta.reg = r.dtb.reg
                = q.dta.reg == type_t::s32 ? type_t::s16 : type_t::s8;
        return r;
    }));
}

} // namespace rvjit
