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

#include <initializer_list>

#include "gtest/gtest.h"

#include "isa_any_registers.hpp"
#include "rvjit/implementation/common/isa/buffer/allocator/group.hpp"
#include "rvjit/implementation/common/isa/buffer/allocator/named.hpp"

namespace rvjit {

using namespace isa_any;

namespace {

/// Builds a named_t from a list of live registers
buffer::allocator::named_t make_named(std::initializer_list<reg_t> regs) {
    name_t ids[buffer::buffer_t::MAXN];
    int n = 0;
    for (const reg_t &r : regs)
        ids[n++] = reg_name(r);
    buffer::allocator::named_t p;
    p.configure(buffer::buffer_t(ids, n));
    return p;
}

/// Builds a group_t over names 0..31 with the given group sizes
buffer::allocator::group_t make_group(int in_gs, int out_gs = -1) {
    buffer::allocator::group_t g;
    g.configure(buffer::buffer_t::from_mask(~0u), in_gs,
            out_gs < 0 ? in_gs : out_gs);
    return g;
}

} // namespace

// Allocation advances the cursor by the right amount

/// name() advances the cursor by exactly one
TEST(buffer_allocator_named, single_alloc) {
    auto p = make_named({r10, r11, r12, r13});
    ASSERT_EQ(p.allocated(), 0);
    p.pop();
    ASSERT_EQ(p.allocated(), 1);
}

/// buffer(n) advances the cursor by exactly `n`
TEST(buffer_allocator_named, panel_alloc) {
    auto p = make_named({r10, r11, r12, r13});
    p.pop(3);
    ASSERT_EQ(p.allocated(), 3);
}

/// buffer(n) advances the cursor by `n` across a larger candidate list
TEST(buffer_allocator_named, block_alloc) {
    auto p = make_named({r10, r11, r12, r13, r15, r16, r17, r8, r9, r18});
    p.pop(6);
    ASSERT_EQ(p.allocated(), 6);
}

/// Successive name()/buffer(n) calls accumulate on the same cursor
TEST(buffer_allocator_named, mixed_alloc) {
    auto p = make_named({r10, r11, r12, r13});
    p.pop(); // 1
    p.pop(2); // +2 = 3
    ASSERT_EQ(p.allocated(), 3);
}

/// allocation_t is ignored: output draws like input
TEST(buffer_allocator_named, ignores_allocation_kind) {
    auto p = make_named({r10, r11, r12, r13});
    p.pop(buffer::allocation_t::output); // 1
    ASSERT_EQ(p.allocated(), 1);
    {
        const int in = p.available(buffer::allocation_t::input);
        const int out = p.available(buffer::allocation_t::output);
        ASSERT_EQ(in, out);
    }
}

// named_t: failed allocations leave the cursor and block intact

/// name() past the candidate list's end fails silently
TEST(buffer_allocator_named, failed_single_alloc) {
    auto p = make_named({r10});
    p.pop(); // consume the only register
    ASSERT_EQ(p.pop(), no_name);
    ASSERT_EQ(p.allocated(), 1);
}

/// buffer(n) beyond availability fails, returning an empty block
TEST(buffer_allocator_named, failed_multi_alloc) {
    auto p = make_named({r10, r11});
    const auto blk = p.pop(3); // needs 3, only 2 available
    ASSERT_EQ(blk.size(), 0);
    ASSERT_EQ(p.allocated(), 0);
}

/// named_t allocator configure() rejects group sizes above 1
TEST(buffer_allocator_named, configure_rejects_group_size_above_one) {
    buffer::allocator::named_t p;
    const reg_t regs[] = {r10, r11, r12, r13};
    ASSERT_FALSE(p.configure(buffer::buffer_t(regs, regs + 4), 2, 2));
}

/// may_reserve() checks that `out + in` still fits the shared budget
TEST(buffer_allocator_named, may_reserve_sums_budget) {
    auto p = make_named({r10, r11, r12, r13});
    EXPECT_TRUE(p.may_reserve(2, 2));
    EXPECT_FALSE(p.may_reserve(3, 2));
}

// group cross-invalidation

/// An output group blocks the input groups it overlaps
TEST(buffer_allocator_group, acc_blocks_inp) {
    auto g = make_group(4, 8);
    // Name 0 at group size 8 occupies names 0-7: blocks input groups 0 and 4
    EXPECT_EQ(g.pop(buffer::allocation_t::output), 0);
    EXPECT_EQ(g.pop(), 8);
}

/// pop() with no group left returns the sentinel, taking nothing
TEST(buffer_allocator_group, failed_single_alloc) {
    auto g = make_group(8);
    for (int i = 0; i < 4; ++i)
        ASSERT_NE(g.pop(), no_name);
    EXPECT_EQ(g.pop(), no_name);
    EXPECT_EQ(g.allocated(), 4);
}

/// An input group blocks the output group it overlaps
TEST(buffer_allocator_group, inp_blocks_acc) {
    auto g = make_group(4, 8);
    g.pop(buffer::allocation_t::output); // names 0-7
    EXPECT_EQ(g.pop(), 8); // the first free input group
    // The output group at 8 is blocked
    EXPECT_EQ(g.pop(buffer::allocation_t::output), 16);
}

/// Single-view inputs and outputs yield the same names
TEST(buffer_allocator_group, single_view_fallback) {
    auto a = make_group(4);
    auto b = make_group(4);
    constexpr int N = 4;
    const buffer::buffer_t acc = a.pop(N, buffer::allocation_t::output);
    const buffer::buffer_t inp = b.pop(N);
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(acc.id(i), inp.id(i)) << "mismatch at i=" << i;
}

// configure(in_gs, out_gs): out_gs must be >= in_gs

/// An output group smaller than the input one is rejected
TEST(buffer_allocator_group, configure_rejects_smaller_output_group) {
    buffer::allocator::group_t g;
    ASSERT_FALSE(g.configure(buffer::buffer_t::from_mask(~0u), 4, 2));
    EXPECT_EQ(g.available(), 0);
}

/// may_reserve() sums `out + in` against one shared budget
TEST(buffer_allocator_group, may_reserve_shared_view_sums_budget) {
    auto g = make_group(8); // 4 candidates (0, 8, 16, 24)
    EXPECT_TRUE(g.may_reserve(2, 2));
    EXPECT_FALSE(g.may_reserve(3, 2));
}

/// may_reserve() checks mixed views against independent budgets
TEST(buffer_allocator_group, may_reserve_mixed_view_checks_independently) {
    auto g = make_group(4, 8); // 8 input candidates, 4 output
    EXPECT_TRUE(g.may_reserve(4, 8));
    EXPECT_FALSE(g.may_reserve(5, 8));
    EXPECT_FALSE(g.may_reserve(4, 9));
}

} // namespace rvjit
