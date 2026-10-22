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

#include <vector>

#include "gtest/gtest.h"

#include "rvjit/implementation/common/stream/operand/operand.hpp"

namespace rvjit {

namespace {

buffer_map_t ring(const name_t *ids, int n, int rows, int cols) {
    return buffer_map_t(buffer::buffer_t(ids, n), {rows, cols});
}

// Records the register of every zeroed accumulator position
struct recording_arithmetic_t : public operation::arithmetic_t {
    bool zero(const arithmetic::op_t &op) const override {
        zeroed.push_back(op.rd.name);
        types.push_back(op.rd.type);
        return true;
    }

    mutable std::vector<name_t> zeroed;
    mutable std::vector<type_t> types;
};

rvjit::operand_t f32_traits() {
    rvjit::operand_t t;
    t.set_type(type_t::f32);
    return t;
}

} // namespace

/// A buffered view resolves registers through the block's ring
TEST(operand_stream, buffered_resolves_registers_through_the_ring) {
    common::stream::operand::operand_t os;
    const name_t ids[] = {3, 4, 5, 6};
    auto *h = os.buffered(ring(ids, 4, 2, 2));
    ASSERT_TRUE(h->is_valid());
    EXPECT_EQ(h->ndims(), 2);
    EXPECT_EQ(h->get({0, 0}), 3);
    EXPECT_EQ(h->get({1, 1}), 6);
}

/// resize() reshapes the view in place over the same registers
TEST(operand_stream, resize_reshapes_in_place) {
    common::stream::operand::operand_t os;
    const name_t ids[] = {3, 4, 5, 6};
    auto *h = os.buffered(ring(ids, 4, 2, 2));
    ASSERT_TRUE(h->is_valid());
    ASSERT_TRUE(h->resize({1, 2}));
    EXPECT_EQ(h->size(0), 1);
    EXPECT_EQ(h->size(1), 2);
    EXPECT_EQ(h->get({0, 1}), 4);
    ASSERT_TRUE(h->resize({2, 2}));
    EXPECT_EQ(h->size(0), 2);
    EXPECT_EQ(h->get({1, 1}), 6);
}

/// A default-constructed stream is not valid
TEST(operand_stream, unconfigured_references_are_invalid) {
    EXPECT_FALSE(stream::ref_t().is_valid());
}

/// A reference drives its stream in the phase it was taken in
TEST(operand_stream, ref_reports_validity_and_shape) {
    common::stream::operand::operand_t os;
    const name_t ids[] = {3, 4};
    auto *h = os.buffered(ring(ids, 2, 1, 2));
    ASSERT_TRUE(h->is_valid());
    stream::ref_t r = h->ref();
    EXPECT_TRUE(r.is_valid());
    EXPECT_EQ(r.size(1), 2);
    EXPECT_EQ(r.get({0, 1}), 4);
}

/// An accumulator's init phase zeroes every register once, in order
TEST(operand_stream, accumulator_init_zeroes_every_register) {
    recording_arithmetic_t arith;
    common::stream::operand::operand_t os(arith);
    const name_t ids[] = {3, 4, 5, 6};
    auto *h = os.accumulator(ring(ids, 4, 2, 2), f32_traits());
    ASSERT_TRUE(h->is_valid());
    h->emit_init();
    const std::vector<name_t> expected = {3, 4, 5, 6};
    EXPECT_EQ(arith.zeroed, expected);
    for (type_t t : arith.types)
        EXPECT_EQ(t, type_t::f32);
}

/// An accumulator zeroes only the registers of its resized shape
TEST(operand_stream, accumulator_init_follows_the_resized_shape) {
    recording_arithmetic_t arith;
    common::stream::operand::operand_t os(arith);
    const name_t ids[] = {3, 4, 5, 6};
    auto *h = os.accumulator(ring(ids, 4, 2, 2), f32_traits());
    ASSERT_TRUE(h->resize({1, 2}));
    h->emit_init();
    const std::vector<name_t> expected = {3, 4};
    EXPECT_EQ(arith.zeroed, expected);
}

/// A buffered view zeroes nothing, even when the factory has arithmetic
TEST(operand_stream, buffered_init_emits_nothing) {
    recording_arithmetic_t arith;
    common::stream::operand::operand_t os(arith);
    const name_t ids[] = {3, 4};
    auto *h = os.buffered(ring(ids, 2, 1, 2), f32_traits());
    h->emit_init();
    EXPECT_TRUE(arith.zeroed.empty());
}

/// An accumulator from a factory without arithmetic zeroes nothing
TEST(operand_stream, accumulator_without_arithmetic_emits_nothing) {
    common::stream::operand::operand_t os;
    const name_t ids[] = {3, 4};
    auto *h = os.accumulator(ring(ids, 2, 1, 2), f32_traits());
    ASSERT_TRUE(h->is_valid());
    h->emit_init();
}

} // namespace rvjit
