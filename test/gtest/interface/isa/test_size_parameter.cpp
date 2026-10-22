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

TEST(size_parameter_t, as_less_eq_refines_kind_and_bound) {
    size_parameter_t p {source_t(r10)};
    EXPECT_FALSE(p.has_bound());
    EXPECT_EQ(p.bound(), 0);

    p.as_less_eq(14);
    EXPECT_TRUE(p.is_less_eq());
    EXPECT_FALSE(p.is_runtime());
    EXPECT_TRUE(p.has_bound());
    EXPECT_EQ(p.bound(), 14);
    EXPECT_EQ(p.value(), 14);
    EXPECT_TRUE(p.source().is_adopt());
}

/// A literal is exactly its value, which is also its bound
TEST(size_parameter_t, literal_is_bounded_by_its_value) {
    const size_parameter_t p = size_parameter_t::literal(5);
    EXPECT_TRUE(p.has_bound());
    EXPECT_FALSE(p.is_less_eq());
    EXPECT_EQ(p.bound(), 5);
}

/// The factories match `size_parameter_t(source_t(r))`, refined by as_less_eq
TEST(size_parameter_t, runtime_and_less_eq_convenience_factories) {
    const size_parameter_t le = size_parameter_t::less_eq(r11, 6);
    EXPECT_TRUE(le.is_less_eq());
    EXPECT_EQ(le.bound(), 6);
    EXPECT_TRUE(le.source().is_adopt());
    EXPECT_TRUE(le.source().buffer() == r11);

    const size_parameter_t rt = size_parameter_t::runtime(r12);
    EXPECT_TRUE(rt.is_runtime());
    EXPECT_FALSE(rt.has_bound());
    EXPECT_TRUE(rt.source().is_adopt());
    EXPECT_TRUE(rt.source().buffer() == r12);
}

} // namespace rvjit
