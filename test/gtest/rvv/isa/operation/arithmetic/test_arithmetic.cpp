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

/// Copies `op`, assigning `d`/`a`/`b` as `rd`/`rs1`/`rs2`'s own names
static arithmetic::op_t with_names(
        arithmetic::op_t op, name_t d, name_t a, name_t b) {
    op.rd.name = d;
    op.rs1.name = a;
    op.rs2.name = b;
    return op;
}

/// Builds a unary `op_t`, assigning `d` as `rd`'s own name
static arithmetic::op_t unary_with_name(rvjit::operand_t rd, name_t d) {
    rd.name = d;
    return arithmetic::op_t(rd);
}

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
        mc.macc(with_names(arithmetic::op_t(op_t::s64, op_t::u8, op_t::s8,
                                   rvjit::form_t::vector),
                d, a, b));
        mc.macc(with_names(
                arithmetic::op_t::widening(op_t::s64, rvjit::form_t::vector), d,
                a, b));
        mc.macc(with_names(arithmetic::op_t(op_t::s32, op_t::s16, op_t::s8,
                                   rvjit::form_t::vector),
                d, a, b));
        mc.macc(with_names(
                arithmetic::op_t::uniform(op_t::bf16, rvjit::form_t::vector), d,
                a, b));
        mc.macc(with_names(arithmetic::op_t(op_t::f16, op_t::f32, op_t::f32,
                                   rvjit::form_t::vector),
                d, a, b));

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
            mc.macc(with_names(
                    arithmetic::op_t::uniform(dt, rvjit::form_t::vector), d, a,
                    b));
            gen.vmacc_vv(vd, vs1, vs2);
        }

        // Widening int
        for (auto dt : {op_t::s8}) {
            mc.macc(with_names(
                    arithmetic::op_t::widening(dt, rvjit::form_t::vector), d, a,
                    b));
            gen.vwmacc_vv(vd, vs1, vs2);
        }
        mc.macc(with_names(
                arithmetic::op_t::widening(op_t::u8, rvjit::form_t::vector), d,
                a, b));
        gen.vwmaccu_vv(vd, vs1, vs2);

        // Uniform float
        for (auto dt : {op_t::f32}) {
            mc.macc(with_names(
                    arithmetic::op_t::uniform(dt, rvjit::form_t::vector), d, a,
                    b));
            gen.vfmacc_vv(vd, vs1, vs2);
        }

        // Widening float
        for (auto dt : {op_t::f16, op_t::bf16}) {
            mc.macc(with_names(
                    arithmetic::op_t::widening(dt, rvjit::form_t::vector), d, a,
                    b));
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
        mc.zero(unary_with_name(
                rvjit::operand_t(rvjit::form_t::scalar, op_t::s32), d));
        gen.mv(rd, x0);

        // Float scalar
        mc.zero(unary_with_name(
                rvjit::operand_t(rvjit::form_t::scalar, op_t::f32), d));
        gen.fmv_d_x(fd, x0);

        // Vector (default form)
        mc.zero(unary_with_name(
                rvjit::operand_t(rvjit::form_t::vector, op_t::f32), d));
        gen.vmv_v_i(vd, 0);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// `supports_*` validates dtype/form combinations without register names
TEST(macc_supports, dtype_combinations) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.supports_macc(op_t::uniform(type_t::f32, vec)));
    {
        const auto op = op_t::uniform(type_t::f32, vec).vector_scalar();
        EXPECT_TRUE(ar.supports_macc(op));
    }
    EXPECT_TRUE(ar.supports_macc(op_t::widening(type_t::f16, vec)));
    EXPECT_TRUE(ar.supports_macc(op_t::widening(type_t::s8, vec)));

    // No 4x widening, no narrowing
    {
        const op_t op(type_t::s32, type_t::s8, type_t::s8, vec);
        EXPECT_FALSE(ar.supports_macc(op));
    }
    {
        const op_t op(type_t::s8, type_t::s32, type_t::s32, vec);
        EXPECT_FALSE(ar.supports_macc(op));
    }

    // `us` widening only exists in the vector-scalar form
    op_t us = op_t::widening(type_t::s8, vec);
    us.rs1 = rvjit::operand_t(vec, type_t::u8);
    EXPECT_FALSE(ar.supports_macc(us));
    EXPECT_TRUE(ar.supports_macc(us.vector_scalar()));

    EXPECT_TRUE(ar.supports_zero(op_t(rvjit::operand_t(vec, type_t::f32))));
    EXPECT_FALSE(ar.supports_macc(op_t(rvjit::operand_t(vec, type_t::f32))));
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
        ar.mul(with_names(vv, d, a, b));
        gen.vmul_vv(vd, vs2, vs1);
        ar.add(with_names(vv, d, a, b));
        gen.vadd_vv(vd, vs2, vs1);

        auto fvv = arithmetic::op_t::uniform(op_t::f64, rvjit::form_t::vector);
        ar.mul(with_names(fvv, d, a, b));
        gen.vfmul_vv(vd, vs2, vs1);
        ar.add(with_names(fvv, d, a, b));
        gen.vfadd_vv(vd, vs2, vs1);

        // Vector-scalar: the scalar is `rs1`
        auto vx = arithmetic::op_t::uniform(op_t::s32, rvjit::form_t::vector)
                          .vector_scalar();
        ar.mul(with_names(vx, d, a, b));
        gen.vmul_vx(vd, vs2, Reg(a));
        ar.add(with_names(vx, d, a, b));
        gen.vadd_vx(vd, vs2, Reg(a));

        auto vf = arithmetic::op_t::uniform(op_t::f32, rvjit::form_t::vector)
                          .vector_scalar();
        ar.mul(with_names(vf, d, a, b));
        gen.vfmul_vf(vd, vs2, FReg(a));
        ar.add(with_names(vf, d, a, b));
        gen.vfadd_vf(vd, vs2, FReg(a));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// mul and add reject widening shapes, unsupported float types and scalars
TEST(mul_add_supports, rejections) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.supports_mul(op_t::uniform(type_t::u8, vec)));
    EXPECT_TRUE(ar.supports_add(op_t::uniform(type_t::f16, vec)));
    EXPECT_FALSE(ar.supports_mul(op_t::widening(type_t::s8, vec)));
    EXPECT_FALSE(ar.supports_add(op_t::uniform(type_t::bf16, vec)));
    EXPECT_FALSE(ar.supports_mul(op_t::uniform(type_t::f32, form_t::scalar)));
    EXPECT_FALSE(ar.supports_add(op_t(rvjit::operand_t(vec, type_t::f32))));
}

/// bitcast moves integer bits into a float register of the same width
TEST(bitcast_emission, widths) {
    static constexpr int n_pairs = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);

        const auto make = [](type_t f, type_t i) {
            arithmetic::op_t op;
            op.rd = rvjit::operand_t(rvjit::form_t::scalar, f, 3);
            op.rs1 = rvjit::operand_t(rvjit::form_t::scalar, i, 5);
            return op;
        };
        ar.bitcast(make(type_t::f16, type_t::s16));
        gen.fmv_h_x(FReg(3), Reg(5));
        ar.bitcast(make(type_t::f32, type_t::s32));
        gen.fmv_w_x(FReg(3), Reg(5));
        ar.bitcast(make(type_t::f64, type_t::s64));
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

    const auto make = [](type_t f, type_t i) {
        arithmetic::op_t op;
        op.rd = rvjit::operand_t(rvjit::form_t::scalar, f);
        op.rs1 = rvjit::operand_t(rvjit::form_t::scalar, i);
        return op;
    };
    EXPECT_TRUE(ar.supports_bitcast(make(type_t::f32, type_t::s32)));
    EXPECT_FALSE(ar.supports_bitcast(make(type_t::f32, type_t::s64)));
    EXPECT_FALSE(ar.supports_bitcast(make(type_t::s32, type_t::s32)));
    EXPECT_FALSE(ar.supports_bitcast(make(type_t::bf16, type_t::s16)));
}

/// narrow emits the float conversions and the saturating integer clips
TEST(narrow_emission, valid_instructions) {
    static constexpr int n_pairs = 7;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src, name_t d, name_t s) {
            ar.narrow(arithmetic::op_t::convert(dst, src, vec).names(d, s));
        };
        emit(type_t::f32, type_t::f64, 8, 16);
        gen.vfncvt_f_f_w(vd, vs);
        emit(type_t::f16, type_t::f32, 8, 16);
        gen.vfncvt_f_f_w(vd, vs);
        emit(type_t::bf16, type_t::f32, 8, 16);
        gen.vfncvtbf16_f_f_w(vd, vs);
        emit(type_t::s8, type_t::s16, 8, 16);
        gen.vnclip_wi(vd, vs, 0);
        emit(type_t::s16, type_t::s32, 8, 16);
        gen.vnclip_wi(vd, vs, 0);
        emit(type_t::u8, type_t::u16, 8, 16);
        gen.vnclipu_wi(vd, vs, 0);

        // In place: the destination is the source's lowest-numbered part
        emit(type_t::f16, type_t::f32, 16, 16);
        gen.vfncvt_f_f_w(vs, vs);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// widen emits the float conversions and the sign or zero extensions
TEST(widen_emission, valid_instructions) {
    static constexpr int n_pairs = 7;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src) {
            ar.widen(arithmetic::op_t::convert(dst, src, vec).names(8, 16));
        };
        emit(type_t::f32, type_t::f16);
        gen.vfwcvt_f_f_v(vd, vs);
        emit(type_t::f64, type_t::f32);
        gen.vfwcvt_f_f_v(vd, vs);
        emit(type_t::f32, type_t::bf16);
        gen.vfwcvtbf16_f_f_v(vd, vs);
        emit(type_t::s16, type_t::s8);
        gen.vwadd_vx(vd, vs, zero);
        emit(type_t::s32, type_t::s16);
        gen.vwadd_vx(vd, vs, zero);
        emit(type_t::u16, type_t::u8);
        gen.vwaddu_vx(vd, vs, zero);
        emit(type_t::s32, type_t::u16);
        gen.vwaddu_vx(vd, vs, zero);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// narrow and widen accept only 2x conversions of one kind
TEST(narrow_widen_supports, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = rvjit::form_t::vector;
    const auto conv = [&](type_t dst, type_t src) {
        return op_t::convert(dst, src, vec);
    };

    EXPECT_TRUE(ar.supports_narrow(conv(type_t::f16, type_t::f32)));
    EXPECT_TRUE(ar.supports_narrow(conv(type_t::s8, type_t::s16)));
    EXPECT_TRUE(ar.supports_narrow(conv(type_t::u8, type_t::u16)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::f32, type_t::f32)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::f16, type_t::f64)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::s8, type_t::s32)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::s16, type_t::f32)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::u16, type_t::s32)));
    EXPECT_FALSE(ar.supports_narrow(conv(type_t::f32, type_t::f16)));
    EXPECT_FALSE(ar.supports_narrow(op_t::uniform(type_t::f32, vec)));
    {
        const auto op = op_t::convert(type_t::f16, type_t::f32, form_t::scalar);
        EXPECT_FALSE(ar.supports_narrow(op));
    }

    EXPECT_TRUE(ar.supports_widen(conv(type_t::f32, type_t::f16)));
    EXPECT_TRUE(ar.supports_widen(conv(type_t::s16, type_t::s8)));
    EXPECT_TRUE(ar.supports_widen(conv(type_t::s32, type_t::u16)));
    EXPECT_FALSE(ar.supports_widen(conv(type_t::f64, type_t::f16)));
    EXPECT_FALSE(ar.supports_widen(conv(type_t::u16, type_t::s8)));
    EXPECT_FALSE(ar.supports_widen(conv(type_t::s16, type_t::f16)));
    EXPECT_FALSE(ar.supports_widen(conv(type_t::f16, type_t::f32)));
}

/// extend emits the 2x, 4x and 8x sign or zero extensions
TEST(extend_emission, valid_instructions) {
    static constexpr int n_pairs = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src) {
            ar.extend(arithmetic::op_t::convert(dst, src, vec).names(8, 16));
        };
        emit(type_t::s16, type_t::s8);
        gen.vsext_vf2(vd, vs);
        emit(type_t::s32, type_t::s8);
        gen.vsext_vf4(vd, vs);
        emit(type_t::s64, type_t::s8);
        gen.vsext_vf8(vd, vs);
        emit(type_t::u16, type_t::u8);
        gen.vzext_vf2(vd, vs);
        emit(type_t::s32, type_t::u8);
        gen.vzext_vf4(vd, vs);
        emit(type_t::s64, type_t::u16);
        gen.vzext_vf4(vd, vs);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// convert emits the vector and scalar integer-float conversions
TEST(convert_emission, valid_instructions) {
    static constexpr int n_pairs = 8;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto sca = rvjit::form_t::scalar;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src, rvjit::form_t f) {
            ar.convert(arithmetic::op_t::convert(dst, src, f).names(8, 16));
        };
        emit(type_t::f32, type_t::s32, vec);
        gen.vfcvt_f_x_v(vd, vs);
        emit(type_t::f16, type_t::u16, vec);
        gen.vfcvt_f_xu_v(vd, vs);
        emit(type_t::s32, type_t::f32, vec);
        gen.vfcvt_x_f_v(vd, vs);
        emit(type_t::u16, type_t::f16, vec);
        gen.vfcvt_xu_f_v(vd, vs);
        emit(type_t::f32, type_t::s32, sca);
        gen.fcvt_s_w(FReg(8), Reg(16));
        emit(type_t::f64, type_t::s64, sca);
        gen.fcvt_d_l(FReg(8), Reg(16));
        emit(type_t::s32, type_t::f16, sca);
        gen.fcvt_w_h(Reg(8), FReg(16));
        emit(type_t::s64, type_t::f32, sca);
        gen.fcvt_l_s(Reg(8), FReg(16));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// extend takes integers by 2, 4 or 8; convert changes kind, vectors at one
/// width
TEST(extend_convert_supports, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = rvjit::form_t::vector;
    const auto sca = rvjit::form_t::scalar;
    const auto conv = [&](type_t dst, type_t src) {
        return op_t::convert(dst, src, vec);
    };

    EXPECT_TRUE(ar.supports_extend(conv(type_t::s32, type_t::s8)));
    EXPECT_TRUE(ar.supports_extend(conv(type_t::s64, type_t::u8)));
    EXPECT_FALSE(ar.supports_extend(conv(type_t::u16, type_t::s8)));
    EXPECT_FALSE(ar.supports_extend(conv(type_t::s8, type_t::s8)));
    EXPECT_FALSE(ar.supports_extend(conv(type_t::f32, type_t::f16)));
    EXPECT_FALSE(
            ar.supports_extend(op_t::convert(type_t::s32, type_t::s8, sca)));

    EXPECT_TRUE(ar.supports_convert(conv(type_t::f32, type_t::s32)));
    EXPECT_TRUE(ar.supports_convert(conv(type_t::s16, type_t::f16)));
    EXPECT_FALSE(ar.supports_convert(conv(type_t::f32, type_t::s16)));
    EXPECT_FALSE(ar.supports_convert(conv(type_t::bf16, type_t::s16)));
    EXPECT_FALSE(ar.supports_convert(conv(type_t::f32, type_t::f32)));
    EXPECT_TRUE(
            ar.supports_convert(op_t::convert(type_t::f32, type_t::s64, sca)));
    EXPECT_FALSE(
            ar.supports_convert(op_t::convert(type_t::f32, type_t::s16, sca)));
}

/// names() renames the leading registers and leaves the other operands alone
TEST(arithmetic_op, names_overrides_the_leading_registers) {
    auto op = arithmetic::op_t::uniform(type_t::f32, form_t::vector);
    op.rd.name = 1;
    op.rs1.name = 2;
    op.rs2.name = 3;

    const auto d = op.names(7);
    EXPECT_EQ(d.rd.name, 7);
    EXPECT_EQ(d.rs1.name, 2);

    const auto ds = op.names(7, 8);
    EXPECT_EQ(ds.rs1.name, 8);
    EXPECT_EQ(ds.rs2.name, 3);

    const auto all = op.names(7, 8, 9);
    EXPECT_EQ(all.rs2.name, 9);
    EXPECT_EQ(op.rd.name, 1);
}

} // namespace rvjit
