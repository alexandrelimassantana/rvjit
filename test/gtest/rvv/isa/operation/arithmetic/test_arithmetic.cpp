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
#include "rvjit/implementation/rvv/isa/operation/arithmetic/arithmetic.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// Invalid dtype/instruction-class combinations emit nothing
TEST(macc_emission, invalid_parameters) {
    using op_t = type_t;

    // Invalid combinations: all emit nothing, leaving only ret()
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t mc(env.e);

        const auto d = 8;
        const auto a = 16;
        const auto b = 24;

        // None of these should emit anything
        mc.macc(arithmetic::op_t(
                op_t::s64, op_t::u8, op_t::s8, rvjit::form_t::vector))(d, a, b);
        mc.macc(arithmetic::op_t::widening(op_t::s64, rvjit::form_t::vector))(
                d, a, b);
        mc.macc(arithmetic::op_t(op_t::s32, op_t::s16, op_t::s8,
                rvjit::form_t::vector))(d, a, b);
        mc.macc(arithmetic::op_t::uniform(op_t::bf16, rvjit::form_t::vector))(
                d, a, b);
        mc.macc(arithmetic::op_t(op_t::f16, op_t::f32, op_t::f32,
                rvjit::form_t::vector))(d, a, b);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_EQ(kernel.getSize(), sizeof(uint32_t));
}

/// Every valid dtype/instruction-class combination emits its matching mnemonic
TEST(macc_emission, valid_instructions) {
    using op_t = type_t;
    static constexpr int n_pairs = 7;

    // Valid combinations: probe and reference instructions interleaved
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t mc(env.e);

        // Common instruction parameters
        const auto d = 8;
        const auto a = 16;
        const auto b = 24;
        const auto vd = VReg(d);
        const auto vs1 = VReg(a);
        const auto vs2 = VReg(b);

        // Uniform int
        for (auto dt : {op_t::s8, op_t::s32}) {
            mc.macc(arithmetic::op_t::uniform(dt, rvjit::form_t::vector))(
                    d, a, b);
            gen.vmacc_vv(vd, vs1, vs2);
        }

        // Widening int
        for (auto dt : {op_t::s8}) {
            mc.macc(arithmetic::op_t::widening(dt, rvjit::form_t::vector))(
                    d, a, b);
            gen.vwmacc_vv(vd, vs1, vs2);
        }
        mc.macc(arithmetic::op_t::widening(op_t::u8, rvjit::form_t::vector))(
                d, a, b);
        gen.vwmaccu_vv(vd, vs1, vs2);

        // Uniform float
        for (auto dt : {op_t::f32}) {
            mc.macc(arithmetic::op_t::uniform(dt, rvjit::form_t::vector))(
                    d, a, b);
            gen.vfmacc_vv(vd, vs1, vs2);
        }

        // Widening float
        for (auto dt : {op_t::f16, op_t::bf16}) {
            mc.macc(arithmetic::op_t::widening(dt, rvjit::form_t::vector))(
                    d, a, b);
            switch (dt) {
                case op_t::bf16: gen.vfwmaccbf16_vv(vd, vs1, vs2); break;
                default: gen.vfwmacc_vv(vd, vs1, vs2); break;
            }
        }

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// zero dispatches to the register type matching op_t::rd()'s own form/kind
TEST(zero_emission, valid_dispatch) {
    using op_t = type_t;
    static constexpr int n_pairs = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t mc(env.e);

        const auto d = 8;
        const auto rd = Reg(d);
        const auto fd = FReg(d);
        const auto vd = VReg(d);

        // Int scalar
        mc.zero(rvjit::form_t::scalar, op_t::s32)(d);
        gen.mv(rd, x0);

        // Float scalar
        mc.zero(rvjit::form_t::scalar, op_t::f32)(d);
        gen.fmv_d_x(fd, x0);

        // Vector (default form)
        mc.zero(rvjit::form_t::vector, op_t::f32)(d);
        gen.vmv_v_i(vd, 0);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// `is_valid` validates dtype/form combinations without registers
TEST(macc_supports, dtype_combinations) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.macc(op_t::uniform(type_t::f32, vec)).is_valid());
    {
        const auto op = op_t::uniform(type_t::f32, vec).vector_scalar();
        EXPECT_TRUE(ar.macc(op).is_valid());
    }
    EXPECT_TRUE(ar.macc(op_t::widening(type_t::f16, vec)).is_valid());
    EXPECT_TRUE(ar.macc(op_t::widening(type_t::s8, vec)).is_valid());

    // No 4x widening, no narrowing
    {
        const op_t op(type_t::s32, type_t::s8, type_t::s8, vec);
        EXPECT_FALSE(ar.macc(op).is_valid());
    }
    {
        const op_t op(type_t::s8, type_t::s32, type_t::s32, vec);
        EXPECT_FALSE(ar.macc(op).is_valid());
    }

    // `us` widening only exists in the vector-scalar form
    op_t us = op_t::widening(type_t::s8, vec);
    us.rs1 = rvjit::operand_t(vec, type_t::u8);
    EXPECT_FALSE(ar.macc(us).is_valid());
    EXPECT_TRUE(ar.macc(us.vector_scalar()).is_valid());

    EXPECT_TRUE(ar.zero(op_t(rvjit::operand_t(vec, type_t::f32))).is_valid());
    EXPECT_FALSE(ar.macc(op_t(rvjit::operand_t(vec, type_t::f32))).is_valid());
}

/// mul and add emit the vector-vector and vector-scalar mnemonics
TEST(mul_add_emission, valid_instructions) {
    using op_t = type_t;
    static constexpr int n_pairs = 8;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);

        const auto d = 8;
        const auto a = 16;
        const auto b = 24;
        const auto vd = VReg(d);
        const auto vs1 = VReg(a);
        const auto vs2 = VReg(b);

        // Vector-vector, int and float
        auto vv = arithmetic::op_t::uniform(op_t::s16, rvjit::form_t::vector);
        ar.mul(vv)(d, a, b);
        gen.vmul_vv(vd, vs2, vs1);
        ar.add(vv)(d, a, b);
        gen.vadd_vv(vd, vs2, vs1);

        auto fvv = arithmetic::op_t::uniform(op_t::f64, rvjit::form_t::vector);
        ar.mul(fvv)(d, a, b);
        gen.vfmul_vv(vd, vs2, vs1);
        ar.add(fvv)(d, a, b);
        gen.vfadd_vv(vd, vs2, vs1);

        // Vector-scalar: the scalar is `rs1`
        auto vx = arithmetic::op_t::uniform(op_t::s32, rvjit::form_t::vector)
                          .vector_scalar();
        ar.mul(vx)(d, a, b);
        gen.vmul_vx(vd, vs2, Reg(a));
        ar.add(vx)(d, a, b);
        gen.vadd_vx(vd, vs2, Reg(a));

        auto vf = arithmetic::op_t::uniform(op_t::f32, rvjit::form_t::vector)
                          .vector_scalar();
        ar.mul(vf)(d, a, b);
        gen.vfmul_vf(vd, vs2, FReg(a));
        ar.add(vf)(d, a, b);
        gen.vfadd_vf(vd, vs2, FReg(a));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// mul and add reject unsupported float types and scalars
TEST(mul_add_supports, rejections) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.mul(op_t::uniform(type_t::u8, vec)).is_valid());
    EXPECT_TRUE(ar.add(op_t::uniform(type_t::f16, vec)).is_valid());
    EXPECT_TRUE(ar.mul(op_t::widening(type_t::s8, vec)).is_valid());
    EXPECT_FALSE(ar.add(op_t::uniform(type_t::bf16, vec)).is_valid());
    EXPECT_FALSE(
            ar.mul(op_t::uniform(type_t::bf16, form_t::scalar)).is_valid());
    EXPECT_FALSE(ar.add(op_t(rvjit::operand_t(vec, type_t::f32))).is_valid());
}

/// mul and add widen at 2x only, never narrow, and reject what RVV lacks
TEST(mul_add_supports, widening_shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.add(op_t::widening(type_t::u16, vec)).is_valid());
    EXPECT_TRUE(ar.add(op_t::widening(type_t::f16, vec)).is_valid());
    EXPECT_TRUE(ar.add(op_t::wide_operand(type_t::s32, vec)).is_valid());
    EXPECT_TRUE(ar.add(op_t::wide_operand(type_t::f32, vec)).is_valid());
    EXPECT_TRUE(ar.mul(op_t::widening(type_t::f32, vec)).is_valid());

    // Narrowing and 4x widening
    const op_t narrowing(type_t::s8, type_t::s16, type_t::s16, vec);
    const op_t quad(type_t::s32, type_t::s8, type_t::s8, vec);
    EXPECT_FALSE(ar.add(narrowing).is_valid());
    EXPECT_FALSE(ar.mul(narrowing).is_valid());
    EXPECT_FALSE(ar.add(quad).is_valid());
    EXPECT_FALSE(ar.mul(quad).is_valid());

    // No bf16 widening add or mul, and no wide-operand mul
    EXPECT_FALSE(ar.add(op_t::widening(type_t::bf16, vec)).is_valid());
    EXPECT_FALSE(ar.mul(op_t::widening(type_t::bf16, vec)).is_valid());
    EXPECT_FALSE(ar.add(op_t::wide_operand(type_t::bf16, vec)).is_valid());
    EXPECT_FALSE(ar.mul(op_t::wide_operand(type_t::s8, vec)).is_valid());

    // A widening add needs sources of one signedness
    const op_t mixed(type_t::s16, type_t::s8, type_t::u8, vec);
    EXPECT_FALSE(ar.add(mixed).is_valid());
    EXPECT_TRUE(ar.mul(mixed).is_valid());

    // A signed scalar times an unsigned vector has no instruction
    const op_t mixed_vx = op_t(mixed).vector_scalar();
    const op_t mixed_su(type_t::s16, type_t::u8, type_t::s8, vec);
    EXPECT_FALSE(ar.mul(mixed_vx).is_valid());
    EXPECT_TRUE(ar.mul(op_t(mixed_su).vector_scalar()).is_valid());
}

/// Widening mul and add emit the widening and wide-operand mnemonics
TEST(mul_add_emission, widening_instructions) {
    using op_t = type_t;
    static constexpr int n_pairs = 18;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;

        const auto d = 8;
        const auto a = 16;
        const auto b = 24;
        const auto vd = VReg(d);
        const auto vs1 = VReg(a);
        const auto vs2 = VReg(b);

        // Widening int add, vector-vector and vector-scalar
        const auto ws = arithmetic::op_t::widening(op_t::s8, vec);
        const auto wu = arithmetic::op_t::widening(op_t::u16, vec);
        ar.add(ws)(d, a, b);
        gen.vwadd_vv(vd, vs2, vs1);
        ar.add(wu)(d, a, b);
        gen.vwaddu_vv(vd, vs2, vs1);
        ar.add(arithmetic::op_t(ws).vector_scalar())(d, a, b);
        gen.vwadd_vx(vd, vs2, Reg(a));
        ar.add(arithmetic::op_t(wu).vector_scalar())(d, a, b);
        gen.vwaddu_vx(vd, vs2, Reg(a));

        // Wide-operand int add
        const auto ows = arithmetic::op_t::wide_operand(op_t::s16, vec);
        const auto owu = arithmetic::op_t::wide_operand(op_t::u8, vec);
        ar.add(ows)(d, a, b);
        gen.vwadd_wv(vd, vs2, vs1);
        ar.add(owu)(d, a, b);
        gen.vwaddu_wv(vd, vs2, vs1);
        ar.add(arithmetic::op_t(ows).vector_scalar())(d, a, b);
        gen.vwadd_wx(vd, vs2, Reg(a));

        // Widening and wide-operand float add
        const auto wf = arithmetic::op_t::widening(op_t::f16, vec);
        const auto owf = arithmetic::op_t::wide_operand(op_t::f32, vec);
        ar.add(wf)(d, a, b);
        gen.vfwadd_vv(vd, vs2, vs1);
        ar.add(arithmetic::op_t(wf).vector_scalar())(d, a, b);
        gen.vfwadd_vf(vd, vs2, FReg(a));
        ar.add(owf)(d, a, b);
        gen.vfwadd_wv(vd, vs2, vs1);
        ar.add(arithmetic::op_t(owf).vector_scalar())(d, a, b);
        gen.vfwadd_wf(vd, vs2, FReg(a));

        // Widening int mul by signedness
        const arithmetic::op_t su(op_t::s16, op_t::u8, op_t::s8, vec);
        const arithmetic::op_t us(op_t::s16, op_t::s8, op_t::u8, vec);
        ar.mul(ws)(d, a, b);
        gen.vwmul_vv(vd, vs2, vs1);
        ar.mul(wu)(d, a, b);
        gen.vwmulu_vv(vd, vs2, vs1);
        ar.mul(su)(d, a, b);
        gen.vwmulsu_vv(vd, vs2, vs1);
        ar.mul(us)(d, a, b);
        gen.vwmulsu_vv(vd, vs1, vs2);
        ar.mul(arithmetic::op_t(ws).vector_scalar())(d, a, b);
        gen.vwmul_vx(vd, vs2, Reg(a));
        ar.mul(arithmetic::op_t(su).vector_scalar())(d, a, b);
        gen.vwmulsu_vx(vd, vs2, Reg(a));

        // Widening float mul
        ar.mul(wf)(d, a, b);
        gen.vfwmul_vv(vd, vs2, vs1);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// div emits `rd = rs2 / rs1` by signedness, the scalar being the divisor
TEST(div_emission, valid_instructions) {
    using op_t = type_t;
    static constexpr int n_pairs = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;

        const auto d = 8;
        const auto a = 16;
        const auto b = 24;
        const auto vd = VReg(d);
        const auto vs1 = VReg(a);
        const auto vs2 = VReg(b);

        const auto s = arithmetic::op_t::uniform(op_t::s32, vec);
        const auto u = arithmetic::op_t::uniform(op_t::u8, vec);
        const auto f = arithmetic::op_t::uniform(op_t::f64, vec);
        ar.div(s)(d, a, b);
        gen.vdiv_vv(vd, vs2, vs1);
        ar.div(arithmetic::op_t(s).vector_scalar())(d, a, b);
        gen.vdiv_vx(vd, vs2, Reg(a));
        ar.div(u)(d, a, b);
        gen.vdivu_vv(vd, vs2, vs1);
        ar.div(arithmetic::op_t(u).vector_scalar())(d, a, b);
        gen.vdivu_vx(vd, vs2, Reg(a));
        ar.div(f)(d, a, b);
        gen.vfdiv_vv(vd, vs2, vs1);
        ar.div(arithmetic::op_t(f).vector_scalar())(d, a, b);
        gen.vfdiv_vf(vd, vs2, FReg(a));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// div takes uniform vector shapes only
TEST(div_supports, rejections) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.div(vec, type_t::f16).is_valid());
    EXPECT_TRUE(ar.div(vec, type_t::u32).is_valid());
    EXPECT_FALSE(ar.div(op_t::widening(type_t::s8, vec)).is_valid());
    EXPECT_FALSE(ar.div(op_t::wide_operand(type_t::s8, vec)).is_valid());
    EXPECT_FALSE(
            ar.div(op_t(type_t::s8, type_t::s16, type_t::s16, vec)).is_valid());
    EXPECT_FALSE(ar.div(vec, type_t::bf16).is_valid());
    EXPECT_FALSE(
            ar.div(op_t::widening(type_t::s32, form_t::scalar)).is_valid());
    EXPECT_FALSE(ar.div(op_t(rvjit::operand_t(vec, type_t::f32))).is_valid());
}

/// Invalid add, mul and div shapes emit nothing
TEST(mul_add_div_emission, invalid_parameters) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        using arithmetic::op_t;

        const op_t narrowing(type_t::s8, type_t::s16, type_t::s16, vec);
        const op_t mixed(type_t::s16, type_t::s8, type_t::u8, vec);
        ar.add(narrowing)(8, 16, 24);
        ar.mul(narrowing)(8, 16, 24);
        ar.div(narrowing)(8, 16, 24);
        ar.add(mixed)(8, 16, 24);
        ar.mul(op_t(mixed).vector_scalar())(8, 16, 24);
        ar.mul(op_t::widening(type_t::bf16, vec))(8, 16, 24);
        ar.div(op_t::widening(type_t::f16, vec))(8, 16, 24);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_EQ(kernel.getSize(), sizeof(uint32_t));
}

/// bitcast moves integer bits into a float register of the same width
TEST(bitcast_emission, widths) {
    static constexpr int n_pairs = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);

        const auto sca = rvjit::form_t::scalar;
        const auto emit = [&](type_t f, type_t i) {
            ar.bitcast({sca, f}, {sca, i})(3, 5);
        };
        emit(type_t::f16, type_t::s16);
        gen.fmv_h_x(FReg(3), Reg(5));
        emit(type_t::f32, type_t::s32);
        gen.fmv_w_x(FReg(3), Reg(5));
        emit(type_t::f64, type_t::s64);
        gen.fmv_d_x(FReg(3), Reg(5));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// bitcast needs an int source and a same-width float destination
TEST(bitcast_supports, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);

    const auto sca = rvjit::form_t::scalar;
    const auto valid = [&](type_t f, type_t i) {
        return ar.bitcast({sca, f}, {sca, i}).is_valid();
    };
    EXPECT_TRUE(valid(type_t::f32, type_t::s32));
    EXPECT_FALSE(valid(type_t::f32, type_t::s64));
    EXPECT_FALSE(valid(type_t::s32, type_t::s32));
    EXPECT_FALSE(valid(type_t::bf16, type_t::s16));
}

/// One expression emits on any registers, each call naming them
TEST(arithmetic_expression, emits_on_the_registers_of_each_call) {
    static constexpr int n_pairs = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto fma = ar.macc(rvjit::form_t::vector, type_t::f32);
        fma(8, 16, 24);
        gen.vfmacc_vv(VReg(8), VReg(16), VReg(24));
        fma(1, 2, 3);
        gen.vfmacc_vv(VReg(1), VReg(2), VReg(3));
        const auto fma_vf = ar.macc({rvjit::form_t::vector, type_t::f32},
                {rvjit::form_t::scalar, type_t::f32},
                {rvjit::form_t::vector, type_t::f32});
        fma_vf(8, 10, 24);
        gen.vfmacc_vf(VReg(8), FReg(10), VReg(24));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// An unbound expression is never valid and emits nothing
TEST(arithmetic_expression, unbound_is_invalid) {
    const arithmetic::expression_t none;
    EXPECT_FALSE(none.is_valid());
}

/// RVV zero clears vectors, and scalars through RV64
TEST(zero_valid, vectors_and_scalars) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    EXPECT_TRUE(ar.zero(rvjit::form_t::vector, type_t::f32).is_valid());
    EXPECT_TRUE(ar.zero(rvjit::form_t::scalar, type_t::s32).is_valid());
    EXPECT_FALSE(ar.zero(rvjit::form_t::undefined, type_t::s32).is_valid());
}

/// RVV mul, add, div and macc emit scalar forms through RV64
TEST(mul_add_div_emission, scalars_through_rv64) {
    static constexpr int n_pairs = 4;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto sca = rvjit::form_t::scalar;

        ar.mul(sca, type_t::f32)(3, 5, 7);
        gen.fmul_s(FReg(3), FReg(5), FReg(7));
        ar.add(sca, type_t::s64)(3, 5, 7);
        gen.add(Reg(3), Reg(5), Reg(7));
        ar.div(sca, type_t::s32)(3, 5, 7);
        gen.divw(Reg(3), Reg(7), Reg(5));
        ar.macc(sca, type_t::f64)(3, 5, 7);
        gen.fmadd_d(FReg(3), FReg(5), FReg(7), FReg(3));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

} // namespace rvjit
