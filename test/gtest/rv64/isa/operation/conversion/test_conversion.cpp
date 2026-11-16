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
#include "rvjit/implementation/rv64/isa/operation/conversion/conversion.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// cast emits the scalar integer-float conversions
TEST(rv64_cast_emission, valid_instructions) {
    static constexpr int n_pairs = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::conversion::conversion_t cv(env.e);
        const auto sca = rvjit::form_t::scalar;

        const auto emit = [&](type_t dst, type_t src) {
            cv.cast(sca, dst, src)(8, 16);
        };
        emit(type_t::f32, type_t::s32);
        gen.fcvt_s_w(FReg(8), Reg(16));
        emit(type_t::f64, type_t::s64);
        gen.fcvt_d_l(FReg(8), Reg(16));
        emit(type_t::s32, type_t::f16);
        gen.fcvt_w_h(Reg(8), FReg(16));
        emit(type_t::s64, type_t::f32);
        gen.fcvt_l_s(Reg(8), FReg(16));
        emit(type_t::f32, type_t::u64);
        gen.fcvt_s_lu(FReg(8), Reg(16));
        emit(type_t::u64, type_t::f64);
        gen.fcvt_lu_d(Reg(8), FReg(16));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// cast takes 32- or 64-bit integers and scalar forms only
TEST(rv64_cast_valid, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    using conversion::op_t;
    const auto sca = rvjit::form_t::scalar;
    const auto vec = rvjit::form_t::vector;

    EXPECT_TRUE(cv.cast(sca, type_t::f32, type_t::s64).is_valid());
    EXPECT_TRUE(cv.cast(sca, type_t::u32, type_t::f64).is_valid());
    EXPECT_FALSE(cv.cast(sca, type_t::f32, type_t::s16).is_valid());
    EXPECT_FALSE(cv.cast(sca, type_t::bf16, type_t::s32).is_valid());
    EXPECT_FALSE(cv.cast(vec, type_t::f32, type_t::s32).is_valid());
}

/// Only cast is implemented: narrow, widen and extend reject everything
TEST(rv64_conversion_valid, only_cast) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    using conversion::op_t;
    const auto sca = rvjit::form_t::scalar;

    EXPECT_FALSE(cv.narrow(sca, type_t::f32, type_t::f64).is_valid());
    EXPECT_FALSE(cv.widen(sca, type_t::f64, type_t::f32).is_valid());
    EXPECT_FALSE(cv.extend(sca, type_t::s64, type_t::s32).is_valid());
}

} // namespace rvjit
