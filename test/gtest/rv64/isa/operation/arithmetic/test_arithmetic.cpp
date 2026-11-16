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
#include "rvjit/implementation/rv64/isa/operation/arithmetic/arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/conversion/conversion.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

namespace {

/// A kernel storing the full register `a2 / a1` of type `dt` at `*a0`,
/// optionally dividing at the type `ext` both operands are extended to first
rv64_jit_probe_t::generate_fn_t div_probe(
        type_t dt, type_t ext = type_t::none) {
    return [dt, ext](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::arithmetic::arithmetic_t ar(env.e);
        rv64::conversion::conversion_t cv(env.e);
        const auto sca = rvjit::form_t::scalar;
        type_t at = dt;
        if (ext != type_t::none) {
            cv.extend(sca, ext, dt)(11, 11);
            cv.extend(sca, ext, dt)(12, 12);
            at = ext;
        }
        ar.div(sca, at)(13, 11, 12);
        gen.sd(Reg(13), Reg(10));
        gen.ret();
    };
}

/// The full register `dividend / divisor` holds, operands extended by `dt`
int64_t divide(type_t dt, int64_t divisor, int64_t dividend,
        type_t ext = type_t::none) {
    rv64_jit_probe_t kernel(div_probe(dt, ext));
    if (!kernel.create_kernel()) return 0;
    int64_t out = 0;
    kernel(&out, divisor, dividend);
    return out;
}

} // namespace

/// Integer mul, add and div pick the word forms at 32 bits, else the
/// doubleword ones, and divide unsigned types with `divu`
TEST(rv64_arithmetic_emission, integer_widths) {
    static constexpr int n_pairs = 15;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::arithmetic::arithmetic_t ar(env.e);
        const auto sca = rvjit::form_t::scalar;

        const auto rd = Reg(5);
        const auto rs1 = Reg(6);
        const auto rs2 = Reg(7);

        ar.add(sca, type_t::s8)(5, 6, 7);
        gen.add(rd, rs1, rs2);
        ar.add(sca, type_t::u16)(5, 6, 7);
        gen.add(rd, rs1, rs2);
        ar.add(sca, type_t::u32)(5, 6, 7);
        gen.addw(rd, rs1, rs2);
        ar.add(sca, type_t::s64)(5, 6, 7);
        gen.add(rd, rs1, rs2);
        ar.mul(sca, type_t::s16)(5, 6, 7);
        gen.mul(rd, rs1, rs2);
        ar.mul(sca, type_t::u32)(5, 6, 7);
        gen.mulw(rd, rs1, rs2);
        ar.mul(sca, type_t::u64)(5, 6, 7);
        gen.mul(rd, rs1, rs2);

        // Division is `rd = rs2 / rs1`
        ar.div(sca, type_t::s8)(5, 6, 7);
        gen.div(rd, rs2, rs1);
        ar.div(sca, type_t::s16)(5, 6, 7);
        gen.div(rd, rs2, rs1);
        ar.div(sca, type_t::s32)(5, 6, 7);
        gen.divw(rd, rs2, rs1);
        ar.div(sca, type_t::s64)(5, 6, 7);
        gen.div(rd, rs2, rs1);
        ar.div(sca, type_t::u8)(5, 6, 7);
        gen.divu(rd, rs2, rs1);
        ar.div(sca, type_t::u16)(5, 6, 7);
        gen.divu(rd, rs2, rs1);
        ar.div(sca, type_t::u32)(5, 6, 7);
        gen.divu(rd, rs2, rs1);
        ar.div(sca, type_t::u64)(5, 6, 7);
        gen.divu(rd, rs2, rs1);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// Float mul, add, div and macc emit their mnemonic by width; macc is the
/// fused `fmadd` accumulating into `rd`
TEST(rv64_arithmetic_emission, float_widths) {
    static constexpr int n_pairs = 11;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::arithmetic::arithmetic_t ar(env.e);
        const auto sca = rvjit::form_t::scalar;

        const auto fd = FReg(5);
        const auto fs1 = FReg(6);
        const auto fs2 = FReg(7);

        ar.add(sca, type_t::f16)(5, 6, 7);
        gen.fadd_h(fd, fs1, fs2);
        ar.add(sca, type_t::f32)(5, 6, 7);
        gen.fadd_s(fd, fs1, fs2);
        ar.add(sca, type_t::f64)(5, 6, 7);
        gen.fadd_d(fd, fs1, fs2);
        ar.mul(sca, type_t::f16)(5, 6, 7);
        gen.fmul_h(fd, fs1, fs2);
        ar.mul(sca, type_t::f64)(5, 6, 7);
        gen.fmul_d(fd, fs1, fs2);
        ar.div(sca, type_t::f16)(5, 6, 7);
        gen.fdiv_h(fd, fs2, fs1);
        ar.div(sca, type_t::f32)(5, 6, 7);
        gen.fdiv_s(fd, fs2, fs1);
        ar.div(sca, type_t::f64)(5, 6, 7);
        gen.fdiv_d(fd, fs2, fs1);
        ar.macc(sca, type_t::f16)(5, 6, 7);
        gen.fmadd_h(fd, fs1, fs2, fd);
        ar.macc(sca, type_t::f32)(5, 6, 7);
        gen.fmadd_s(fd, fs1, fs2, fd);
        ar.macc(sca, type_t::f64)(5, 6, 7);
        gen.fmadd_d(fd, fs1, fs2, fd);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// Extended operands divide into quotients extended the same way
TEST(rv64_div_values, quotients_keep_the_extension) {
    EXPECT_EQ(divide(type_t::s8, 7, -100), -14);
    EXPECT_EQ(divide(type_t::s16, -3, 30000), -10000);
    EXPECT_EQ(divide(type_t::u16, 2, 60000), 30000);
    EXPECT_EQ(divide(type_t::u32, 1, 4000000000), 4000000000);
    EXPECT_EQ(divide(type_t::s32, -1, INT32_MIN), INT32_MIN);
    EXPECT_EQ(divide(type_t::s64, -2, 7), -3);
    EXPECT_EQ(divide(type_t::u64, 2, -1), INT64_MAX);
}

/// The minimum s8 or s16 over -1 and a division by zero are exact in the low
/// bits only; dividing at the next wider type avoids the first
TEST(rv64_div_values, edge_cases_are_exact_in_the_low_bits) {
    const int64_t min8 = divide(type_t::s8, -1, INT8_MIN);
    EXPECT_EQ(min8, 128);
    EXPECT_EQ(static_cast<int8_t>(min8), INT8_MIN);
    const int64_t min16 = divide(type_t::s16, -1, INT16_MIN);
    EXPECT_EQ(static_cast<int16_t>(min16), INT16_MIN);

    const int64_t by_zero = divide(type_t::u8, 0, 200);
    EXPECT_EQ(by_zero, -1);
    EXPECT_EQ(static_cast<uint8_t>(by_zero), UINT8_MAX);

    EXPECT_EQ(divide(type_t::s8, -1, INT8_MIN, type_t::s16), 128);
}

/// Narrow sums are exact in the low bits, and extend restores the rest
TEST(rv64_add_values, narrow_sums_extend_back) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::arithmetic::arithmetic_t ar(env.e);
        rv64::conversion::conversion_t cv(env.e);
        const auto sca = rvjit::form_t::scalar;

        // a0[0] = a1 + a2 as u8, a0[1] = that sum extended to u16
        ar.add(sca, type_t::u8)(13, 11, 12);
        gen.sd(Reg(13), Reg(10));
        cv.extend(sca, type_t::u16, type_t::u8)(13, 13);
        gen.sd(Reg(13), Reg(10), 8);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    int64_t out[2] = {0, 0};
    kernel(out, int64_t(200), int64_t(100));
    EXPECT_EQ(static_cast<uint8_t>(out[0]), 44);
    EXPECT_EQ(out[1], 44);
}

/// Scalar arithmetic takes uniform integer or f16/f32/f64 shapes; macc takes
/// floats only
TEST(rv64_arithmetic_supports, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto sca = form_t::scalar;

    EXPECT_TRUE(ar.add(sca, type_t::s8).is_valid());
    EXPECT_TRUE(ar.mul(sca, type_t::u16).is_valid());
    EXPECT_TRUE(ar.div(sca, type_t::u32).is_valid());
    EXPECT_TRUE(ar.div(sca, type_t::f16).is_valid());
    EXPECT_TRUE(ar.macc(sca, type_t::f32).is_valid());

    // No bf16, integer macc or widening scalar arithmetic
    EXPECT_FALSE(ar.add(sca, type_t::bf16).is_valid());
    EXPECT_FALSE(ar.macc(sca, type_t::bf16).is_valid());
    EXPECT_FALSE(ar.macc(sca, type_t::s32).is_valid());
    EXPECT_FALSE(ar.mul(op_t::widening(type_t::s32, sca)).is_valid());
    EXPECT_FALSE(ar.macc(op_t::widening(type_t::f16, sca)).is_valid());
    EXPECT_FALSE(ar.add(op_t::wide_operand(type_t::f32, sca)).is_valid());

    // Vector operands are left to RVV
    EXPECT_FALSE(ar.add(form_t::vector, type_t::s32).is_valid());
    const op_t vf = op_t::uniform(type_t::f32, sca).vector_scalar();
    EXPECT_FALSE(ar.mul(vf).is_valid());
    EXPECT_FALSE(ar.macc(vf).is_valid());
}

} // namespace rvjit
