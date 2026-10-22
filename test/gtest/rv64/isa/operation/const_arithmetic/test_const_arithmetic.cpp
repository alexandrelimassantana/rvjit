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
#include <tuple>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

namespace {

enum class const_arith_op_t { add, div, mul, align_down };

/// Correctness references for constant arithmetic operations
int64_t get_ref(int base, int constant, const_arith_op_t op) {
    switch (op) {
        case const_arith_op_t::add: return base + constant;
        case const_arith_op_t::div: return base / constant;
        case const_arith_op_t::align_down: return base - (base % constant);
        case const_arith_op_t::mul: return base * constant;
        default: return 0;
    }
}

/// Verifies if the alignment of `rd` and `rs1` is valid for const arithmetics
///
/// @note The alias is illegal on `div`, `mul` and `align_down` unless
///     `is_pow(c)`
bool alias_is_legal(const_arith_op_t op, int c) {
    if (op == const_arith_op_t::add) return true;
    if (!is_simm12(c)) return true;
    if (op == const_arith_op_t::align_down && c == 1) return true;
    const int64_t abs_c = c < 0 ? -static_cast<int64_t>(c) : c;
    return is_pow2(abs_c);
}

} // namespace

/// Expectations concerning resourceless constants representable by simm12
TEST(const_t, plain_imm) {
    for (int v : {0, 2047, -2048}) {
        const const_t c(v);
        EXPECT_TRUE(c.is_imm());
        EXPECT_TRUE(c.has_imm());
        EXPECT_FALSE(c.is_reg());
        EXPECT_FALSE(c.needs_buffer());
        EXPECT_TRUE(c.is_valid());
    }
    EXPECT_TRUE(const_t(0).is_zero());
    EXPECT_TRUE(const_t().is_zero());
    EXPECT_FALSE(const_t(2047).is_zero());
}

/// Expectations concerning resourceless constants not representable by simm12
TEST(const_t, imm_without_reg) {
    for (int v : {2048, -2049}) {
        const const_t c(v);
        EXPECT_FALSE(c.is_imm());
        EXPECT_TRUE(c.has_imm());
        EXPECT_TRUE(c.is_reg());
        EXPECT_TRUE(c.needs_buffer());
        EXPECT_FALSE(c.has_reg());
        EXPECT_FALSE(c.is_valid());
        EXPECT_FALSE(c.is_zero());
        EXPECT_EQ(c.imm(), v);
    }
}

/// Expectations concerning reg-supplied constants not representable by simm12
TEST(const_t, imm_with_reg) {
    const const_t c(2048, Xbyak_riscv::a0);
    EXPECT_FALSE(c.is_imm());
    EXPECT_TRUE(c.has_imm());
    EXPECT_TRUE(c.is_reg());
    EXPECT_FALSE(c.needs_buffer());
    EXPECT_TRUE(c.needs_to_materialize());
    EXPECT_TRUE(c.has_reg());
    EXPECT_TRUE(c.is_valid());
    EXPECT_EQ(c.imm(), 2048);

    // A small value attaches no register even when one is offered
    const const_t small(16, Xbyak_riscv::a0);
    EXPECT_TRUE(small.is_imm());
    EXPECT_FALSE(small.has_reg());
}

/// Expectations concerning constants without compile-time imm values
TEST(const_t, pure_reg) {
    const const_t c(Xbyak_riscv::a0);
    EXPECT_TRUE(c.is_reg());
    EXPECT_TRUE(c.has_reg());
    EXPECT_FALSE(c.is_imm());
    EXPECT_FALSE(c.has_imm());
    EXPECT_FALSE(c.needs_buffer());
    EXPECT_TRUE(c.is_valid());
    EXPECT_FALSE(c.is_zero());

    EXPECT_TRUE(const_t(zero).is_zero());
}

/// Code generator for general const arithmetic kernels
rv64_jit_probe_t::generate_fn_t const_arithmetic_probe(
        int c, const_arith_op_t op, bool alias_rd = false) {
    return [c, op, alias_rd](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg ptr = Xbyak_riscv::a0;
        const Reg rs1 = Xbyak_riscv::a1;
        const Reg tmp = Xbyak_riscv::a2;
        const Reg rd = alias_rd ? rs1 : Xbyak_riscv::a3;
        const const_t cst = const_t(c, tmp);

        env.xfer.init({cst});

        switch (op) {
            case const_arith_op_t::add: {
                env.arith.add(rd, rs1, cst);
                break;
            }
            case const_arith_op_t::div: {
                env.arith.div(rd, rs1, cst);
                break;
            }
            case const_arith_op_t::align_down: {
                env.arith.align_down(rd, rs1, cst);
                break;
            }
            case const_arith_op_t::mul: {
                env.arith.mul(rd, rs1, cst);
                break;
            }
        }
        gen.sd(rd, ptr);
        gen.ret();
    };
}

class const_arithmetic_correctness_t
    : public ::testing::TestWithParam<std::tuple<int64_t, const_arith_op_t>> {};

/// Correctness expectations under proper component API usage
TEST_P(const_arithmetic_correctness_t, matches_reference) {
    static const int64_t base = 4096;

    int c;
    const_arith_op_t op;
    std::tie(c, op) = GetParam();
    rv64_jit_probe_t kernel(const_arithmetic_probe(c, op));
    ASSERT_TRUE(kernel.create_kernel());

    int64_t expected = get_ref(base, c, op);
    int64_t received;
    kernel(&received, base);

    ASSERT_EQ(expected, received);
}

#define OPS \
    ::testing::Values(const_arith_op_t::add, const_arith_op_t::div, \
            const_arith_op_t::align_down, const_arith_op_t::mul)

/// Positive, power of two, within simm12 range
INSTANTIATE_TEST_SUITE_P(positive_pow2_simm12, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(1, 1024), OPS));

/// Positive, power of two, overflows simm12
INSTANTIATE_TEST_SUITE_P(positive_pow2_reg, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(2048, 4096), OPS));

/// Positive, not a power of two, within simm12 range
INSTANTIATE_TEST_SUITE_P(positive_nonpow2_simm12,
        const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(3, 2047), OPS));

/// Positive, not a power of two, overflows simm12
INSTANTIATE_TEST_SUITE_P(positive_nonpow2_reg, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(2049), OPS));

/// Negative, power of two, within simm12 range
INSTANTIATE_TEST_SUITE_P(negative_pow2_simm12, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(-1, -2048), OPS));

/// Negative, power of two, overflows simm12
INSTANTIATE_TEST_SUITE_P(negative_pow2_reg, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(-4096), OPS));

/// Negative, not a power of two, within simm12 range
INSTANTIATE_TEST_SUITE_P(negative_nonpow2_simm12,
        const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(-3, -2047), OPS));

/// Negative, not a power of two, overflows simm12
INSTANTIATE_TEST_SUITE_P(negative_nonpow2_reg, const_arithmetic_correctness_t,
        ::testing::Combine(::testing::Values(-2049), OPS));

#undef OPS

class const_arithmetic_alias_t
    : public ::testing::TestWithParam<std::tuple<int64_t, const_arith_op_t>> {};

/// Aliasing `rd` and `rs1`: invalid cases are no-ops, valid ones correct
TEST_P(const_arithmetic_alias_t, alias_rd_rs1) {
    static const int64_t base = 4096;

    int c;
    const_arith_op_t op;
    std::tie(c, op) = GetParam();
    const bool alias_rd = true;
    rv64_jit_probe_t kernel(const_arithmetic_probe(c, op, alias_rd));
    ASSERT_TRUE(kernel.create_kernel());

    int64_t expected = alias_is_legal(op, c) ? get_ref(base, c, op) : base;
    int64_t received;
    kernel(&received, base);

    ASSERT_EQ(expected, received);
}

#define ROUND_OPS \
    ::testing::Values(const_arith_op_t::div, const_arith_op_t::align_down, \
            const_arith_op_t::mul)

/// Invalid aliasing
INSTANTIATE_TEST_SUITE_P(noop, const_arithmetic_alias_t,
        ::testing::Combine(::testing::Values(3, -3, 2047, -2047), ROUND_OPS));

/// Valid alias
INSTANTIATE_TEST_SUITE_P(emit, const_arithmetic_alias_t,
        ::testing::Combine(
                ::testing::Values(1, -1, 1024, -2048, 2048, -4096, 2049, -2049),
                ROUND_OPS));

#undef ROUND_OPS

class const_arithmetic_and_t : public ::testing::TestWithParam<int> {};

/// and_ masks with an immediate, or with the register an oversized one needs
TEST_P(const_arithmetic_and_t, matches_reference) {
    static const int64_t base = 0x13579bdf;
    const int c = GetParam();

    rv64_jit_probe_t kernel([c](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg ptr = Xbyak_riscv::a0;
        const Reg rs1 = Xbyak_riscv::a1;
        const const_t cst = const_t(c, Xbyak_riscv::a2);
        env.xfer.init({cst});
        env.arith.and_(Xbyak_riscv::a3, rs1, cst);
        gen.sd(Xbyak_riscv::a3, ptr);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    int64_t received;
    kernel(&received, base);
    ASSERT_EQ(base & c, received);
}

INSTANTIATE_TEST_SUITE_P(masks, const_arithmetic_and_t,
        ::testing::Values(1, 4, 0x7ff, 0x800, 4096, 0x12345, -1, -2048));

} // namespace rvjit
