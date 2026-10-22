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
#include "rvjit/interface/isa/source.hpp"

namespace rvjit {

using namespace isa_any;

/// Every kind is valid when given a real origin, and reports only its own kind
TEST(source_t, valid_origins) {
    EXPECT_TRUE(source_t::none().is_none());
    EXPECT_TRUE(source_t::none().is_valid());

    const source_t imm = source_t::immediate(42);
    EXPECT_TRUE(imm.is_immediate());
    EXPECT_EQ(imm.immediate(), 42);
    EXPECT_TRUE(imm.is_valid());

    const source_t adopt = source_t::adopt(r10);
    EXPECT_TRUE(adopt.is_adopt());
    EXPECT_FALSE(adopt.is_copy());
    EXPECT_TRUE(adopt.buffer() == r10);
    EXPECT_TRUE(adopt.is_valid());

    const source_t copy = source_t::copy(r10);
    EXPECT_TRUE(copy.is_copy());
    EXPECT_TRUE(copy.is_valid());

    const source_t load = source_t::load(pointer_t(r10, 8));
    EXPECT_TRUE(load.is_load());
    EXPECT_TRUE(load.memory().base == r10);
    EXPECT_EQ(load.memory().offset, 8);
    EXPECT_TRUE(load.is_valid());
}

/// A default-constructed source is the invalid sentinel, apart from `none()`
TEST(source_t, default_is_the_invalid_sentinel) {
    const source_t s;
    EXPECT_TRUE(s.is_invalid());
    EXPECT_FALSE(s.is_none());
    EXPECT_FALSE(s.is_valid());
    EXPECT_FALSE(s.is_lazy());

    EXPECT_FALSE(source_t::none().is_invalid());
    EXPECT_TRUE(source_t::none().is_valid());
    EXPECT_TRUE(source_t::none().to_copy().is_none());
    EXPECT_TRUE(source_t().to_copy().is_invalid());
}

/// A zero register or zero-base pointer is rejected, never auto-allocated
TEST(source_t, zero_register_is_invalid) {
    EXPECT_FALSE(source_t::adopt(reg_t()).is_valid());
    EXPECT_FALSE(source_t::copy(reg_t()).is_valid());
}

TEST(source_t, load_with_a_real_pointer_is_valid) {
    const source_t ref = source_t::load(pointer_t(r10, 8));
    EXPECT_TRUE(ref.is_load());
    EXPECT_TRUE(ref.memory().base == r10);
    EXPECT_EQ(ref.memory().offset, 8);
    EXPECT_TRUE(ref.is_valid());
}

TEST(source_t, load_with_zero_base_pointer_is_invalid) {
    const source_t ref = source_t::load(pointer_t());
    EXPECT_TRUE(ref.is_load());
    EXPECT_FALSE(ref.is_valid());
}

/// Only `immediate`, `copy` and `load` imply an emission when materialized
TEST(source_t, is_lazy_covers_the_emitting_kinds) {
    EXPECT_FALSE(source_t::none().is_lazy());
    EXPECT_FALSE(source_t::adopt(r10).is_lazy());
    EXPECT_TRUE(source_t::immediate(1).is_lazy());
    EXPECT_TRUE(source_t::copy(r10).is_lazy());
    EXPECT_TRUE(source_t::load(pointer_t(r10, 0)).is_lazy());
}

/// `to_copy()` turns an `adopt` into a `copy` and keeps every other kind
TEST(source_t, to_copy_reinterprets_adopt_only) {
    const source_t b = source_t::adopt(r10).to_copy();
    EXPECT_TRUE(b.is_copy());
    EXPECT_TRUE(b.buffer() == r10);

    EXPECT_TRUE(source_t::copy(r10).to_copy().is_copy());
    EXPECT_TRUE(source_t::immediate(1).to_copy().is_immediate());
    EXPECT_TRUE(source_t::load(pointer_t(r10, 8)).to_copy().is_load());
    EXPECT_TRUE(source_t::none().to_copy().is_none());
}

} // namespace rvjit
