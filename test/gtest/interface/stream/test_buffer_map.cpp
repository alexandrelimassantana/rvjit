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

#include "rvjit/interface/stream/buffer_map.hpp"

namespace rvjit {

namespace {

const name_t ids[] = {10, 11, 12, 13, 14, 15, 16, 17};

buffer::buffer_t regs(int n) {
    return buffer::buffer_t(ids, n);
}

} // namespace

/// A map built from a shape is a row-major ring over its storage
TEST(buffer_map, shape_constructor_is_a_row_major_ring) {
    buffer_map_t m(regs(6), {2, 3});
    ASSERT_TRUE(m.is_valid());
    EXPECT_EQ(m.total(), 6);
    EXPECT_EQ(m({0, 0}), 10);
    EXPECT_EQ(m({0, 2}), 12);
    EXPECT_EQ(m({1, 0}), 13);
    EXPECT_EQ(m({1, 2}), 15);
}

/// A leading axis past its extent wraps around the ring
TEST(buffer_map, leading_axis_wraps_around_the_ring) {
    buffer_map_t m(regs(4), {2, 2});
    EXPECT_EQ(m({2, 0}), m({0, 0}));
    EXPECT_EQ(m({3, 1}), m({1, 1}));
}

/// A layout override reorders positions over the same storage
TEST(buffer_map, layout_overrides_the_position_order) {
    buffer_map_t m(regs(6), {3, 2});
    m.layout({1, 3}, 6);
    EXPECT_EQ(m({0, 1}), 13);
    EXPECT_EQ(m({2, 1}), 15);
}

/// A transposed map packs the first axis fastest and is as dense as default
TEST(buffer_map, transposed_order_is_dense) {
    buffer_map_t m(regs(6), {3, 2}, order_t::transposed);
    EXPECT_EQ(m.stride[0], 1);
    EXPECT_EQ(m.stride[1], 3);
    EXPECT_EQ(m.modulus, 6);
    EXPECT_EQ(m({2, 1}), 15);
}

/// A wrapped map repacks over a reshape and keeps its depth
TEST(buffer_map, wrapped_map_repacks_when_reshaped) {
    buffer_map_t m = buffer_map_t(regs(3), {4, 2}).wrapped(3);
    EXPECT_EQ(m.total(), 3);
    EXPECT_EQ(m.stride[0], 2);
    m.reshape({4, 1});
    EXPECT_EQ(m.stride[0], 1);
    EXPECT_EQ(m.modulus, 3);
    EXPECT_EQ(m({3, 0}), m({0, 0}));

    // Maps that were not wrapped keep their layout
    buffer_map_t fixed(regs(8), {4, 2});
    fixed.reshape({4, 1});
    EXPECT_EQ(fixed.stride[0], 2);
}

/// A layout given by its strides alone wraps at the extent they address
TEST(buffer_map, layout_defaults_the_modulus_to_the_extent) {
    buffer_map_t m(regs(6), {3, 2});
    m.layout({1, 3});
    EXPECT_EQ(m.modulus, 6);
    EXPECT_EQ(m({2, 1}), 15);

    buffer_map_t row(regs(4), {1, 4});
    row.layout({0, 1});
    EXPECT_EQ(row.modulus, 4);
}

/// A flat modulus smaller than the storage wraps the whole offset
TEST(buffer_map, flat_modulus_wraps_across_axes) {
    buffer_map_t m(regs(2), {4, 3});
    m.layout({3, 1}, 2);
    EXPECT_EQ(m({0, 0}), 10);
    EXPECT_EQ(m({0, 1}), 11);
    EXPECT_EQ(m({0, 2}), 10);
    EXPECT_EQ(m({1, 0}), 11);
}

/// A narrow ring of one row repeats its columns past the ring width
TEST(buffer_map, narrow_ring_wraps_trailing_axis_at_width_one_row) {
    buffer_map_t m(regs(3), {1, 3});
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(m({k, 4}), 11) << "k=" << k;
}

/// A map is invalid until its storage covers every addressed register
TEST(buffer_map, invalid_without_enough_storage) {
    EXPECT_FALSE(buffer_map_t().is_valid());
    EXPECT_FALSE(buffer_map_t({2, 2}).is_valid());
    EXPECT_FALSE(buffer_map_t(regs(3), {2, 2}).is_valid());
    EXPECT_TRUE(buffer_map_t(regs(4), {2, 2}).is_valid());
}

/// A zero extent never resolves, so no position can divide by zero
TEST(buffer_map, zero_shape_is_invalid) {
    EXPECT_FALSE(buffer_map_t(regs(4), {0, 2}).is_valid());
    EXPECT_FALSE(buffer_map_t(regs(4), {2, 0}).is_valid());
}

} // namespace rvjit
