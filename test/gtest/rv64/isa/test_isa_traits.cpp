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

#include "rvjit/implementation/rv64/isa_traits.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

namespace {

// A kernel deriving from the code generator, as a caller writes one
struct derived_kernel_t : public codegen_t {
    int extra = 0;
};

} // namespace

/// An interface handle built from a derived kernel recovers its generator
TEST(rv64_isa_traits, emitter_recovers_a_derived_code_generator) {
    derived_kernel_t kernel;
    const rvjit::emitter_t e(kernel);
    const codegen_t *expected = &kernel;
    EXPECT_TRUE(e.is_attached());
    EXPECT_EQ(e.as<codegen_t>(), expected);
    EXPECT_EQ(&rv64::emitter_t(e).cg(), expected);
    EXPECT_EQ(rvjit::emitter_t().as<codegen_t>(), nullptr);
}

/// The zero register and the stack pointer take the reserved names
TEST(rv64_isa_traits, special_registers_take_reserved_names) {
    EXPECT_EQ(reg_t(zero), reg_t::zero());
    EXPECT_EQ(reg_t(sp), reg_t::stack());
    EXPECT_EQ(reg_t(a0).name(), 10);
    EXPECT_EQ(reg_name(a0), 10);
    EXPECT_FALSE(reg_t(zero).is_not_zero());
    EXPECT_TRUE(reg_t(sp).is_not_zero());
}

/// Every name converts back to the native register it came from
TEST(rv64_isa_traits, names_convert_back_to_native_registers) {
    EXPECT_TRUE(Reg(reg_t(a0)) == a0);
    EXPECT_TRUE(Reg(reg_t::zero()) == zero);
    EXPECT_TRUE(Reg(reg_t::stack()) == sp);
    EXPECT_TRUE(Reg(reg_t()) == zero);
    EXPECT_TRUE(FReg(freg_t(fa0)) == fa0);
    EXPECT_TRUE(VReg(vreg_t(v8)) == v8);
    EXPECT_EQ(freg_t(ft0).name(), 0);
}

/// A predicate maps to the vector mask operand
TEST(rv64_isa_traits, predicate_maps_to_the_mask_operand) {
    using base_t = rv64::isa_traits_base_t;
    EXPECT_TRUE(base_t::vm(predicate_t::masked) == VM::masked);
    EXPECT_TRUE(base_t::vm(predicate_t::unmasked) == VM::unmasked);
}

/// Only the `s` registers are callee-saved, reserved names never are
TEST(rv64_isa_traits, callee_saved_registers_are_the_s_registers) {
    EXPECT_TRUE(is_callee_saved_gpr(reg_name(s0)));
    EXPECT_TRUE(is_callee_saved_gpr(reg_name(s11)));
    EXPECT_FALSE(is_callee_saved_gpr(reg_name(a0)));
    EXPECT_FALSE(is_callee_saved_gpr(stack_name));
}

} // namespace rvjit
