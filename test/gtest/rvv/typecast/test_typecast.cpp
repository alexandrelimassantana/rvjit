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

#include <string>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/rvv/isa/operation/conversion/conversion.hpp"
#include "rvjit/implementation/rvv/typecast/typecast.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

namespace {

using step_t = typecast::typecast_t::step_t;

/// The vtype every test runs at: f32 accumulators in groups of four
rvv::vtype_t f32_m4() {
    rvv::vtype_t vt;
    vt.for_uniform(SEW::e32, LMUL::m4);
    return vt;
}

/// A vector f32 operand, the state hint of the tests
operand_t f32_hint() {
    return operand_t(form_t::vector, type_t::f32);
}

/// The planned steps of each pair, and the pairs no chain converts
TEST(typecast_configure, steps) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    const auto vec = form_t::vector;
    const auto kinds = [&](type_t from, type_t to) {
        rvv::typecast::typecast_t tc(cv, env.e, vt);
        std::string s;
        if (!tc.configure(conversion::op_t(to, from, vec), f32_hint()))
            return std::string("-");
        for (int i = 0; i < tc.nsteps(); ++i) {
            const step_t::kind_t k = tc.step(i).kind;
            s += k == step_t::extend     ? 'E'
                    : k == step_t::widen ? 'W'
                    : k == step_t::cast  ? 'C'
                                         : 'N';
        }
        return s;
    };
    EXPECT_EQ(kinds(type_t::f32, type_t::f32), "");
    EXPECT_EQ(kinds(type_t::f16, type_t::f32), "W");
    EXPECT_EQ(kinds(type_t::f32, type_t::bf16), "N");
    EXPECT_EQ(kinds(type_t::s16, type_t::s32), "W");
    EXPECT_EQ(kinds(type_t::s8, type_t::s32), "E");
    EXPECT_EQ(kinds(type_t::s32, type_t::f32), "C");
    EXPECT_EQ(kinds(type_t::u8, type_t::f32), "EC");
    EXPECT_EQ(kinds(type_t::f32, type_t::s8), "CNN");
    EXPECT_EQ(kinds(type_t::f16, type_t::s8), "CN");
    EXPECT_EQ(kinds(type_t::s32, type_t::s8), "NN");
    EXPECT_EQ(kinds(type_t::f16, type_t::f64), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::u8), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::u32), "-");
    EXPECT_EQ(kinds(type_t::u64, type_t::f64), "C");
    EXPECT_EQ(kinds(type_t::u32, type_t::u64), "W");
    EXPECT_EQ(kinds(type_t::s64, type_t::u64), "-");
    EXPECT_EQ(kinds(type_t::u8, type_t::s8), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::f16), "-");
    EXPECT_EQ(kinds(type_t::bf16, type_t::s16), "-");
}

/// Each step chains in place from the source operand to the destination one
TEST(typecast_configure, steps_chain_the_operands) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const conversion::op_t op(type_t::s8, type_t::f32, form_t::vector);
    ASSERT_TRUE(tc.configure(op, f32_hint()));
    ASSERT_EQ(tc.nsteps(), 3);
    EXPECT_EQ(tc.step(0).op.src.type, type_t::f32);
    EXPECT_EQ(tc.step(0).op.dst.type, type_t::s32);
    EXPECT_EQ(tc.step(1).op.src.type, type_t::s32);
    EXPECT_EQ(tc.step(2).op.dst.type, type_t::s8);
    EXPECT_EQ(tc.src().type, type_t::f32);
    EXPECT_EQ(tc.dst().type, type_t::s8);
    EXPECT_EQ(tc.ratio(), 4);
}

/// Each position runs at the width of its step, the exit at the hint's
TEST(typecast_configure, positions_run_at_their_widths) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const auto vec = form_t::vector;

    // A widening runs at its source's width, and returns
    ASSERT_TRUE(tc.configure(
            conversion::op_t(type_t::f32, type_t::f16, vec), f32_hint()));
    EXPECT_TRUE(tc.interposes(0));
    EXPECT_EQ(tc.config(0).sew, SEW::e16);
    EXPECT_EQ(tc.config(0).lmul, LMUL::m2);
    EXPECT_TRUE(tc.interposes(1));
    EXPECT_EQ(tc.config(1).sew, SEW::e32);
    EXPECT_EQ(tc.config(1).lmul, LMUL::m4);

    // An extension and a cast run at the destination's width, the hint's
    ASSERT_TRUE(tc.configure(
            conversion::op_t(type_t::f32, type_t::u8, vec), f32_hint()));
    EXPECT_EQ(tc.sew(0), SEW::e32);
    EXPECT_FALSE(tc.interposes(0));
    EXPECT_FALSE(tc.interposes(1));
    EXPECT_FALSE(tc.interposes(2));

    // Each narrowing halves the width, the exit returns to the hint's
    ASSERT_TRUE(tc.configure(
            conversion::op_t(type_t::s8, type_t::f32, vec), f32_hint()));
    EXPECT_FALSE(tc.interposes(0));
    EXPECT_TRUE(tc.interposes(1));
    EXPECT_EQ(tc.config(1).lmul, LMUL::m2);
    EXPECT_TRUE(tc.interposes(2));
    EXPECT_EQ(tc.config(2).sew, SEW::e8);
    EXPECT_EQ(tc.config(2).lmul, LMUL::m1);
    EXPECT_TRUE(tc.interposes(3));
    EXPECT_EQ(tc.config(3).sew, SEW::e32);
}

/// A non-empty RVV typecast needs a vector state hint; an empty one does not
TEST(typecast_configure, requires_a_vector_state_hint) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const auto vec = form_t::vector;
    const conversion::op_t widen(type_t::f32, type_t::f16, vec);
    const operand_t scalar(form_t::scalar, type_t::f32);

    EXPECT_FALSE(tc.configure(widen));
    EXPECT_FALSE(tc.is_valid());
    EXPECT_FALSE(tc.configure(widen, scalar));
    EXPECT_TRUE(tc.configure(conversion::op_t(type_t::f32, type_t::f32, vec)));
    EXPECT_TRUE(tc.is_valid());
    EXPECT_TRUE(tc.empty());
}

/// A position narrower than the group takes the fractional LMUL keeping the
/// vector length
TEST(typecast_configure, takes_fractional_groups) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    rvv::vtype_t vt;
    vt.for_uniform(SEW::e64, LMUL::m1);
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const conversion::op_t op(type_t::s8, type_t::s64, form_t::vector);
    ASSERT_TRUE(tc.configure(op, operand_t(form_t::vector, type_t::s64)));
    EXPECT_EQ(tc.config(0).lmul, LMUL::mf2);
    EXPECT_EQ(tc.config(1).lmul, LMUL::mf4);
    EXPECT_EQ(tc.config(2).lmul, LMUL::mf8);
}

/// A position whose LMUL would exceed m8 is rejected
TEST(typecast_configure, rejects_groups_beyond_m8) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    rvv::vtype_t vt;
    vt.for_uniform(SEW::e8, LMUL::m2);
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const conversion::op_t op(type_t::s64, type_t::s8, form_t::vector);
    EXPECT_FALSE(tc.configure(op, operand_t(form_t::vector, type_t::s8)));
    EXPECT_FALSE(tc.is_valid());
    EXPECT_TRUE(tc.empty());
}

/// A failed configuration leaves the typecast empty and invalid
TEST(typecast_configure, a_failed_configure_is_empty) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const auto vec = form_t::vector;
    ASSERT_TRUE(tc.configure(
            conversion::op_t(type_t::f32, type_t::f16, vec), f32_hint()));
    EXPECT_FALSE(tc.configure(
            conversion::op_t(type_t::u32, type_t::s32, vec), f32_hint()));
    EXPECT_TRUE(tc.empty());
    EXPECT_FALSE(tc.is_valid());
    EXPECT_FALSE(tc.interposes(0));
}

/// A scalar typecast casts through RV64, with no state hint and no switch
TEST(typecast_configure, scalar_casts_need_no_state) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::conversion::conversion_t cv(env.e);
    const rvv::vtype_t vt = f32_m4();
    rvv::typecast::typecast_t tc(cv, env.e, vt);
    const auto sca = form_t::scalar;
    ASSERT_TRUE(tc.configure(conversion::op_t(type_t::f32, type_t::s32, sca)));
    EXPECT_EQ(tc.nsteps(), 1);
    EXPECT_FALSE(tc.interposes(0));
    EXPECT_FALSE(tc.interposes(1));
    ASSERT_TRUE(tc.configure(conversion::op_t(type_t::f32, type_t::s8, sca)));
    EXPECT_EQ(tc.nsteps(), 2);
    EXPECT_FALSE(tc.interposes(1));
    ASSERT_TRUE(tc.configure(conversion::op_t(type_t::s32, type_t::s16, sca)));
    EXPECT_EQ(tc.nsteps(), 1);
    EXPECT_FALSE(tc.configure(conversion::op_t(type_t::s16, type_t::s32, sca)));
}

/// Interposing and emitting a store's narrowing emits each step at its width
TEST(typecast_emission, narrowing_switches_widths) {
    static constexpr int n_pairs = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const rvv::vtype_t vt = f32_m4();
        rvv::typecast::typecast_t tc(cv, env.e, vt);
        const conversion::op_t op(type_t::s8, type_t::f32, form_t::vector);
        tc.configure(op, f32_hint());
        const VReg v(8);

        tc.emit(0, 8, 8);
        gen.vfcvt_x_f_v(v, v);
        tc.interpose(1);
        gen.vsetvli(x0, x0, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
        tc.emit(1, 8, 8);
        gen.vnclip_wi(v, v, 0);
        tc.interpose(2);
        gen.vsetvli(x0, x0, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        tc.emit(2, 8, 8);
        gen.vnclip_wi(v, v, 0);
        tc.interpose(3);
        gen.vsetvli(x0, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// emit runs every step of one value with its interpositions, back to the hint
TEST(typecast_emission, emit_runs_the_whole_chain) {
    static constexpr int n_instructions = 6;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::conversion::conversion_t cv(env.e);
        const rvv::vtype_t vt = f32_m4();
        rvv::typecast::typecast_t tc(cv, env.e, vt);
        const conversion::op_t op(type_t::s8, type_t::f32, form_t::vector);
        tc.configure(op, f32_hint());
        const VReg v(8);

        // The probe, then the reference instructions
        tc.emit(8, 8);
        gen.vfcvt_x_f_v(v, v);
        gen.vsetvli(x0, x0, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
        gen.vnclip_wi(v, v, 0);
        gen.vsetvli(x0, x0, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
        gen.vnclip_wi(v, v, 0);
        gen.vsetvli(x0, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_halves(kernel, n_instructions);
}

} // namespace
} // namespace rvjit
