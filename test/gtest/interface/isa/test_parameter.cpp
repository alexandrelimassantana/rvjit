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
#include "rvjit/interface/isa/parameter.hpp"

namespace rvjit {

using namespace isa_any;

/// A default parameter_t has a `none` reference and resolves to `runtime`
TEST(parameter_t, default_constructed_is_runtime_with_no_reference) {
    const parameter_t p;
    EXPECT_TRUE(p.is_runtime());
    EXPECT_TRUE(p.source().is_none());
}

/// A constant source_t always resolves to `literal`
TEST(parameter_t, constant_reference_resolves_to_literal) {
    const parameter_t p = parameter_t::literal(7);
    EXPECT_TRUE(p.is_literal());
    EXPECT_EQ(p.value(), 7);
    EXPECT_TRUE(p.source().is_immediate());
    EXPECT_EQ(p.source().immediate(), 7);

    const parameter_t from_ref(source_t::immediate(9));
    EXPECT_TRUE(from_ref.is_literal());
    EXPECT_EQ(from_ref.value(), 9);
}

/// A non-constant source_t resolves to an unbounded `runtime` until refined
TEST(parameter_t, non_constant_reference_resolves_to_runtime) {
    const parameter_t from_buffer {source_t(r10)};
    EXPECT_TRUE(from_buffer.is_runtime());
    EXPECT_TRUE(from_buffer.source().is_adopt());

    const parameter_t from_memory {source_t(pointer_t(r10, 0))};
    EXPECT_TRUE(from_memory.is_runtime());
    EXPECT_TRUE(from_memory.source().is_load());

    const parameter_t from_none {source_t::none()};
    EXPECT_TRUE(from_none.is_runtime());
    EXPECT_TRUE(from_none.source().is_none());
}

TEST(parameter_t, is_same_value_requires_equal_literals) {
    const parameter_t a = parameter_t::literal(7);
    EXPECT_TRUE(a.is_same_value(parameter_t::literal(7)));
    EXPECT_FALSE(a.is_same_value(parameter_t::literal(8)));
    EXPECT_FALSE(a.is_same_value(parameter_t(source_t(r10))));
    {
        const parameter_t p {source_t(r10)};
        EXPECT_FALSE(p.is_same_value(parameter_t(source_t(r10))));
    }
}

/// A single type is the same in memory and in registers
TEST(type_parameter_t, one_type_never_converts) {
    const type_parameter_t t(type_t::s8);
    EXPECT_FALSE(t.converts());
    EXPECT_FALSE(t.widens());
    EXPECT_FALSE(t.narrows());
    EXPECT_EQ(t.ratio(), 1);
    EXPECT_EQ(t.offset(4), 0);
}

/// A widening value is read from the top `1 / ratio` of a register group
TEST(type_parameter_t, widening_reads_the_top_of_a_group) {
    const type_parameter_t t(type_t::s8, type_t::s32);
    EXPECT_TRUE(t.converts());
    EXPECT_TRUE(t.widens());
    EXPECT_EQ(t.ratio(), 4);
    EXPECT_EQ(t.offset(4), 3);
    EXPECT_EQ(t.offset(8), 6);
    EXPECT_EQ(type_parameter_t(type_t::f16, type_t::f32).offset(2), 1);
}

/// Narrowing and same-width conversions have no widening factor
TEST(type_parameter_t, narrowing_and_same_width_have_no_ratio) {
    const type_parameter_t load(type_t::f16, type_t::f32);
    const type_parameter_t store(type_t::f32, type_t::f16);
    EXPECT_TRUE(store.narrows());
    EXPECT_EQ(store.ratio(), 1);
    EXPECT_EQ(store.offset(4), 0);
    const type_parameter_t retype(type_t::s32, type_t::f32);
    EXPECT_TRUE(retype.converts());
    EXPECT_FALSE(retype.widens() || retype.narrows());
    EXPECT_EQ(retype.offset(4), 0);
    EXPECT_TRUE(load != store);
}

} // namespace rvjit
