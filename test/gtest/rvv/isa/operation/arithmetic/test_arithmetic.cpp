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

/// mul and add reject widening shapes, unsupported float types and scalars
TEST(mul_add_supports, rejections) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    using arithmetic::op_t;
    const auto vec = form_t::vector;

    EXPECT_TRUE(ar.mul(op_t::uniform(type_t::u8, vec)).is_valid());
    EXPECT_TRUE(ar.add(op_t::uniform(type_t::f16, vec)).is_valid());
    EXPECT_FALSE(ar.mul(op_t::widening(type_t::s8, vec)).is_valid());
    EXPECT_FALSE(ar.add(op_t::uniform(type_t::bf16, vec)).is_valid());
    EXPECT_FALSE(ar.mul(op_t::uniform(type_t::f32, form_t::scalar)).is_valid());
    EXPECT_FALSE(ar.add(op_t(rvjit::operand_t(vec, type_t::f32))).is_valid());
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

/// narrow emits the float conversions and the saturating integer clips
TEST(narrow_emission, valid_instructions) {
    static constexpr int n_pairs = 7;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::arithmetic::arithmetic_t ar(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src, name_t d, name_t s) {
            ar.narrow(arithmetic::op_t::convert(dst, src, vec))(d, s);
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
            ar.widen(arithmetic::op_t::convert(dst, src, vec))(8, 16);
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

    EXPECT_TRUE(ar.narrow(conv(type_t::f16, type_t::f32)).is_valid());
    EXPECT_TRUE(ar.narrow(conv(type_t::s8, type_t::s16)).is_valid());
    EXPECT_TRUE(ar.narrow(conv(type_t::u8, type_t::u16)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::f32, type_t::f32)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::f16, type_t::f64)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::s8, type_t::s32)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::s16, type_t::f32)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::u16, type_t::s32)).is_valid());
    EXPECT_FALSE(ar.narrow(conv(type_t::f32, type_t::f16)).is_valid());
    EXPECT_FALSE(ar.narrow(op_t::uniform(type_t::f32, vec)).is_valid());
    {
        const auto op = op_t::convert(type_t::f16, type_t::f32, form_t::scalar);
        EXPECT_FALSE(ar.narrow(op).is_valid());
    }

    EXPECT_TRUE(ar.widen(conv(type_t::f32, type_t::f16)).is_valid());
    EXPECT_TRUE(ar.widen(conv(type_t::s16, type_t::s8)).is_valid());
    EXPECT_TRUE(ar.widen(conv(type_t::s32, type_t::u16)).is_valid());
    EXPECT_FALSE(ar.widen(conv(type_t::f64, type_t::f16)).is_valid());
    EXPECT_FALSE(ar.widen(conv(type_t::u16, type_t::s8)).is_valid());
    EXPECT_FALSE(ar.widen(conv(type_t::s16, type_t::f16)).is_valid());
    EXPECT_FALSE(ar.widen(conv(type_t::f16, type_t::f32)).is_valid());
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
            ar.extend(arithmetic::op_t::convert(dst, src, vec))(8, 16);
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
            ar.convert(arithmetic::op_t::convert(dst, src, f))(8, 16);
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

    EXPECT_TRUE(ar.extend(conv(type_t::s32, type_t::s8)).is_valid());
    EXPECT_TRUE(ar.extend(conv(type_t::s64, type_t::u8)).is_valid());
    EXPECT_FALSE(ar.extend(conv(type_t::u16, type_t::s8)).is_valid());
    EXPECT_FALSE(ar.extend(conv(type_t::s8, type_t::s8)).is_valid());
    EXPECT_FALSE(ar.extend(conv(type_t::f32, type_t::f16)).is_valid());
    EXPECT_FALSE(
            ar.extend(op_t::convert(type_t::s32, type_t::s8, sca)).is_valid());

    EXPECT_TRUE(ar.convert(conv(type_t::f32, type_t::s32)).is_valid());
    EXPECT_TRUE(ar.convert(conv(type_t::s16, type_t::f16)).is_valid());
    EXPECT_FALSE(ar.convert(conv(type_t::f32, type_t::s16)).is_valid());
    EXPECT_FALSE(ar.convert(conv(type_t::bf16, type_t::s16)).is_valid());
    EXPECT_FALSE(ar.convert(conv(type_t::f32, type_t::f32)).is_valid());
    EXPECT_TRUE(ar.convert(op_t::convert(type_t::f32, type_t::s64, sca))
                        .is_valid());
    EXPECT_FALSE(ar.convert(op_t::convert(type_t::f32, type_t::s16, sca))
                         .is_valid());
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

} // namespace rvjit
