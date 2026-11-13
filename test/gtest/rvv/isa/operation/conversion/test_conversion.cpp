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
#include "rvjit/implementation/rvv/isa/operation/conversion/conversion.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// narrow emits the float conversions and the saturating integer clips
TEST(narrow_emission, valid_instructions) {
    static constexpr int n_pairs = 7;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src, name_t d, name_t s) {
            cv.narrow(vec, dst, src)(d, s);
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
        rvv::conversion::conversion_t cv(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src) {
            cv.widen(vec, dst, src)(8, 16);
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
TEST(narrow_widen_valid, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    using conversion::op_t;
    const auto vec = rvjit::form_t::vector;
    const auto narrow = [&](type_t dst, type_t src) {
        return cv.narrow(vec, dst, src).is_valid();
    };
    const auto widen = [&](type_t dst, type_t src) {
        return cv.widen(vec, dst, src).is_valid();
    };

    EXPECT_TRUE(narrow(type_t::f16, type_t::f32));
    EXPECT_TRUE(narrow(type_t::s8, type_t::s16));
    EXPECT_TRUE(narrow(type_t::u8, type_t::u16));
    EXPECT_FALSE(narrow(type_t::f32, type_t::f32));
    EXPECT_FALSE(narrow(type_t::f16, type_t::f64));
    EXPECT_FALSE(narrow(type_t::s8, type_t::s32));
    EXPECT_FALSE(narrow(type_t::s16, type_t::f32));
    EXPECT_FALSE(narrow(type_t::u16, type_t::s32));
    EXPECT_FALSE(narrow(type_t::f32, type_t::f16));
    EXPECT_FALSE(cv.narrow(op_t()).is_valid());

    EXPECT_TRUE(widen(type_t::f32, type_t::f16));
    EXPECT_TRUE(widen(type_t::s16, type_t::s8));
    EXPECT_TRUE(widen(type_t::s32, type_t::u16));
    EXPECT_FALSE(widen(type_t::f64, type_t::f16));
    EXPECT_FALSE(widen(type_t::u16, type_t::s8));
    EXPECT_FALSE(widen(type_t::s16, type_t::f16));
    EXPECT_FALSE(widen(type_t::f16, type_t::f32));
}

/// extend emits the 2x, 4x and 8x sign or zero extensions
TEST(extend_emission, valid_instructions) {
    static constexpr int n_pairs = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src) {
            cv.extend(vec, dst, src)(8, 16);
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

/// cast emits the vector integer-float conversions, and the scalar ones
/// through RV64
TEST(cast_emission, valid_instructions) {
    static constexpr int n_pairs = 8;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const auto vec = rvjit::form_t::vector;
        const auto sca = rvjit::form_t::scalar;
        const auto vd = VReg(8), vs = VReg(16);

        const auto emit = [&](type_t dst, type_t src, rvjit::form_t f) {
            cv.cast(f, dst, src)(8, 16);
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

/// extend takes integers by 2, 4 or 8; cast changes kind at one width
TEST(extend_cast_valid, shapes) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    using conversion::op_t;
    const auto vec = rvjit::form_t::vector;
    const auto extend = [&](type_t dst, type_t src) {
        return cv.extend(vec, dst, src).is_valid();
    };
    const auto cast = [&](type_t dst, type_t src) {
        return cv.cast(vec, dst, src).is_valid();
    };

    EXPECT_TRUE(extend(type_t::s32, type_t::s8));
    EXPECT_TRUE(extend(type_t::s64, type_t::u8));
    EXPECT_FALSE(extend(type_t::u16, type_t::s8));
    EXPECT_FALSE(extend(type_t::s8, type_t::s8));
    EXPECT_FALSE(extend(type_t::f32, type_t::f16));

    EXPECT_TRUE(cast(type_t::f32, type_t::s32));
    EXPECT_TRUE(cast(type_t::s16, type_t::f16));
    EXPECT_FALSE(cast(type_t::f32, type_t::s16));
    EXPECT_FALSE(cast(type_t::bf16, type_t::s16));
    EXPECT_FALSE(cast(type_t::f32, type_t::f32));
}

/// Scalar forms are RV64's: only the cast is valid; mixed forms never are
TEST(conversion_valid, scalar_forms_follow_rv64) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    using conversion::op_t;
    const auto sca = rvjit::form_t::scalar;
    const auto vec = rvjit::form_t::vector;

    EXPECT_FALSE(cv.narrow(sca, type_t::f16, type_t::f32).is_valid());
    EXPECT_FALSE(cv.widen(sca, type_t::f32, type_t::f16).is_valid());
    EXPECT_FALSE(cv.extend(sca, type_t::s32, type_t::s8).is_valid());
    EXPECT_TRUE(cv.cast(sca, type_t::f32, type_t::s32).is_valid());
    EXPECT_TRUE(cv.cast(sca, type_t::f32, type_t::s64).is_valid());
    EXPECT_FALSE(cv.cast(sca, type_t::f32, type_t::s16).is_valid());

    const op_t mixed(operand_t(vec, type_t::f32), operand_t(sca, type_t::s32));
    EXPECT_FALSE(cv.cast(mixed).is_valid());
}

/// An invalid conversion emits nothing
TEST(conversion_emission, invalid_emits_nothing) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const auto sca = rvjit::form_t::scalar;
        const auto vec = rvjit::form_t::vector;

        cv.cast(sca, type_t::f32, type_t::s16)(8, 16);
        cv.narrow(vec, type_t::f32, type_t::f32)(8, 16);
        cv.extend(vec, type_t::u16, type_t::s8)(8, 16);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_EQ(kernel.getSize(), sizeof(uint32_t));
}

/// One expression converts any registers, each call naming them
TEST(conversion_expression, emits_on_the_registers_of_each_call) {
    static constexpr int n_pairs = 2;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const auto f2h
                = cv.narrow(rvjit::form_t::vector, type_t::f16, type_t::f32);
        f2h(8, 16);
        gen.vfncvt_f_f_w(VReg(8), VReg(16));
        f2h(4, 4);
        gen.vfncvt_f_f_w(VReg(4), VReg(4));

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// A conversion's predicates follow its operands' widths and kinds
TEST(conversion_op, predicates_follow_the_operands) {
    const conversion::op_t op(type_t::f16, type_t::f32, form_t::vector);
    EXPECT_TRUE(op.is_narrowing());
    EXPECT_FALSE(op.is_widening());
    EXPECT_FALSE(op.is_casting());
    EXPECT_TRUE(op.is_float());
}

} // namespace rvjit
