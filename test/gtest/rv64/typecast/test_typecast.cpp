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
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/rv64/typecast/typecast.hpp"

namespace rvjit {

namespace {

/// A scalar cast is one step that interposes nothing, with no state hint
TEST(rv64_typecast, casts_without_interposing) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    rv64::typecast::typecast_t tc(cv);
    const conversion::op_t op(type_t::f32, type_t::s32, form_t::scalar);
    ASSERT_TRUE(tc.configure(op));
    ASSERT_EQ(tc.nsteps(), 1);
    EXPECT_EQ(tc.step(0).kind, typecast::typecast_t::step_t::cast);
    EXPECT_FALSE(tc.interposes(0));
    EXPECT_FALSE(tc.interposes(1));
}

/// Chains needing a narrowing or a float widening are rejected
TEST(rv64_typecast, rejects_unimplemented_steps) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    rv64::typecast::typecast_t tc(cv);
    const auto sca = form_t::scalar;
    EXPECT_FALSE(tc.configure(conversion::op_t(type_t::s16, type_t::s32, sca)));
    EXPECT_FALSE(tc.configure(conversion::op_t(type_t::f64, type_t::f32, sca)));
    EXPECT_FALSE(tc.configure(conversion::op_t(type_t::s8, type_t::f32, sca)));
}

/// A 2x integer widening plans one extend step, as the scalar extension
/// accepts a doubling
TEST(rv64_typecast, extends_integers) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    rv64::typecast::typecast_t tc(cv);
    const conversion::op_t op(type_t::s32, type_t::s16, form_t::scalar);
    ASSERT_TRUE(tc.configure(op));
    ASSERT_EQ(tc.nsteps(), 1);
    EXPECT_EQ(tc.step(0).kind, typecast::typecast_t::step_t::extend);
}

/// Vector operands are rejected: every RV64 step is scalar-to-scalar
TEST(rv64_typecast, rejects_vector_forms) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rv64::conversion::conversion_t cv(env.e);
    rv64::typecast::typecast_t tc(cv);
    const conversion::op_t op(type_t::f32, type_t::s32, form_t::vector);
    EXPECT_FALSE(tc.configure(op));
}

/// emit runs the whole chain, writing the cast into the float register
TEST(rv64_typecast, emit_runs_the_whole_chain) {
    static constexpr int n_pairs = 1;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::conversion::conversion_t cv(env.e);
        rv64::typecast::typecast_t tc(cv);
        tc.configure(
                conversion::op_t(type_t::f32, type_t::s32, form_t::scalar));
        tc.emit(8, 16);
        gen.fcvt_s_w(FReg(8), Reg(16));
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// A narrow integer extends in place in its integer register, then casts
/// into the float one
TEST(rv64_typecast, extends_then_casts) {
    static constexpr int n_instructions = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::conversion::conversion_t cv(env.e);
        rv64::typecast::typecast_t tc(cv);
        tc.configure(conversion::op_t(type_t::f32, type_t::s8, form_t::scalar));
        tc.emit(8, 16);
        gen.slli(Reg(16), Reg(16), 56);
        gen.srai(Reg(16), Reg(16), 56);
        gen.fcvt_s_w(FReg(8), Reg(16));
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_halves(kernel, n_instructions);
}

/// emit casts through the dialect
TEST(rv64_typecast, emits_the_cast) {
    static constexpr int n_pairs = 1;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rv64::conversion::conversion_t cv(env.e);
        rv64::typecast::typecast_t tc(cv);
        tc.configure(
                conversion::op_t(type_t::f64, type_t::s64, form_t::scalar));
        tc.emit(0, 8, 16);
        gen.fcvt_d_l(FReg(8), Reg(16));
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

} // namespace
} // namespace rvjit
