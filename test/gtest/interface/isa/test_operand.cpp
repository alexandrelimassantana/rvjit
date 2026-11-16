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

#include "rvjit/interface/isa/operand.hpp"

namespace rvjit {

/// u64 is an unsigned 8-byte integer with no wider natural type
TEST(operand_type, u64_traits) {
    const operand_t o(form_t::scalar, type_t::u64);
    EXPECT_TRUE(o.is_int());
    EXPECT_FALSE(o.is_signed());
    EXPECT_EQ(o.width, 8);
    EXPECT_EQ(sewb(type_t::u64), sewb(type_t::s64));
    EXPECT_EQ(natural_wide(type_t::u64), type_t::none);
    EXPECT_EQ(natural_wide(type_t::u32), type_t::s64);
}

} // namespace rvjit
