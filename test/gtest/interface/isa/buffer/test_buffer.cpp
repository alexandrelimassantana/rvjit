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
#include "rvjit/implementation/common/isa/buffer/allocator/named.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"

namespace rvjit {

using namespace isa_any;

TEST(buffer_t, id_enumerates_names_in_ascending_order) {
    const buffer::buffer_t b({r20, r10, r13});
    ASSERT_EQ(b.size(), 3);
    EXPECT_EQ(b.id(0), reg_name(r10));
    EXPECT_EQ(b.id(1), reg_name(r13));
    EXPECT_EQ(b.id(2), reg_name(r20));
}

TEST(buffer_t, id_out_of_range_is_no_name) {
    const buffer::buffer_t b({r10, r13});
    EXPECT_EQ(b.id(2), no_name);
    EXPECT_EQ(b.id(-1), no_name);
    EXPECT_EQ(buffer::buffer_t().id(0), no_name);
}

TEST(buffer_t, duplicates_collapse_and_filter_removes) {
    const buffer::buffer_t b({r10, r10, r11});
    EXPECT_EQ(b.size(), 2);
    const buffer::buffer_t f = b.filter(buffer::buffer_t({r10}));
    ASSERT_EQ(f.size(), 1);
    EXPECT_EQ(f.id(0), reg_name(r11));
    EXPECT_TRUE(buffer::buffer_t().is_empty());
}

TEST(buffer_t, ordered_follows_order_then_ascending) {
    buffer::order_t order;
    order.push_back(reg_name(r11));
    order.push_back(reg_name(r10));
    const buffer::buffer_t b({r10, r11, r5});
    const buffer::order_t r = b.ordered(order);
    ASSERT_EQ(r.size(), 3);
    EXPECT_EQ(r[0], reg_name(r11));
    EXPECT_EQ(r[1], reg_name(r10));
    EXPECT_EQ(r[2], reg_name(r5));
}

TEST(buffer_t, named_allocator_honors_the_given_order) {
    buffer::order_t order;
    order.push_back(reg_name(r11));
    order.push_back(reg_name(r10));
    buffer::allocator::named_t alloc;
    ASSERT_TRUE(alloc.configure(buffer::buffer_t({r10, r11}), order));
    EXPECT_EQ(alloc.pop(), reg_name(r11));
    EXPECT_EQ(alloc.pop(), reg_name(r10));
}

} // namespace rvjit
