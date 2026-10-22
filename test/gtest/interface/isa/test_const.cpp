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

#include "isa_any_registers.hpp"
#include "rvjit/interface/isa/const.hpp"

namespace rvjit {

using namespace isa_any;

/// An immediate scales by its factor, a factor of 1 keeps any constant
TEST(const_t, scaled_immediate) {
    const const_t c = const_t(8).scaled(4);
    EXPECT_TRUE(c.is_imm());
    EXPECT_EQ(c.imm(), 32);

    EXPECT_TRUE(const_t(r10).scaled(1).has_reg());
}

/// A register-sourced constant cannot scale
TEST(const_t, scaled_fails_when_register_sourced) {
    EXPECT_TRUE(const_t(r10).scaled(2).source().is_invalid());
    EXPECT_FALSE(const_t(r10).scaled(2).is_valid());
    EXPECT_TRUE(const_t(1024).scaled(1).source().is_immediate());
}

/// A product past simm12 needs a register to be valid
TEST(const_t, scaled_overflow_needs_a_register) {
    const_t c = const_t(1024).scaled(4);
    EXPECT_TRUE(c.is_imm_reg());
    EXPECT_EQ(c.imm(), 4096);
    EXPECT_FALSE(c.has_reg());
    EXPECT_EQ(c.buffer_needed(), 1);
    EXPECT_FALSE(c.is_valid());

    c.set_buffer(r11);
    EXPECT_TRUE(c.has_reg());
    EXPECT_TRUE(c.is_valid());
}

/// A failed scale stays invalid when a register is supplied
TEST(const_t, scaled_failure_stays_invalid_with_a_register) {
    const_t c = const_t(r10).scaled(2);
    c.set_buffer(r11);
    EXPECT_TRUE(c.has_reg());
    EXPECT_FALSE(c.is_valid());
}

/// The zero register is a valid register constant even though it has no buffer
TEST(const_t, zero_register_constant_is_valid) {
    const const_t c {zero};
    EXPECT_TRUE(c.source().is_none());
    EXPECT_FALSE(c.has_reg());
    EXPECT_TRUE(c.is_valid());
}

} // namespace rvjit
