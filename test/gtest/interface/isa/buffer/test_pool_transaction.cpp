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

#include "isa_any_fixture.hpp"
#include "gtest/gtest.h"

#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/buffered.hpp"

namespace rvjit {

namespace {

/// Needs a fixed number of buffers
struct wants_t : buffered_t {
    explicit wants_t(int n) : n_(n) {}
    int buffer_needed() const override {
        const int left = n_ - buffer_.size();
        return left > 0 ? left : 0;
    }
    const buffer::buffer_t &buffer() const { return buffer_; }

private:
    int n_;
};

} // namespace

struct buffer_transaction : isa_any_pool_test_t {};
struct buffer_allocation : isa_any_pool_test_t {};

// buffer::buffer_pool_t transactions

/// drop() restores every allocator, and the same registers come back
TEST_F(buffer_transaction, drop_restores_all_allocators) {
    pool.set_vector_group_size(1);
    const int ni = pool.integers().available();
    const int nf = pool.floats().available();
    const int nv = pool.vectors().available();

    ASSERT_TRUE(pool.begin());
    const buffer::buffer_t drawn = pool.integers().pop(3);
    pool.floats().pop(2);
    pool.vectors().pop(4);
    ASSERT_TRUE(pool.drop());

    EXPECT_EQ(pool.integers().available(), ni);
    EXPECT_EQ(pool.floats().available(), nf);
    EXPECT_EQ(pool.vectors().available(), nv);
    EXPECT_EQ(pool.integers().pop(3), drawn);
}

/// commit() keeps the allocations
TEST_F(buffer_transaction, commit_keeps_allocations) {
    const int ni = pool.integers().available();

    ASSERT_TRUE(pool.begin());
    pool.integers().pop(3);
    ASSERT_TRUE(pool.commit());

    EXPECT_EQ(pool.integers().available(), ni - 3);
}

/// An outer drop reverts allocations an inner commit had kept
TEST_F(buffer_transaction, nested_commit_then_outer_drop) {
    const int ni = pool.integers().available();

    ASSERT_TRUE(pool.begin());
    pool.integers().pop(1);
    ASSERT_TRUE(pool.begin());
    pool.integers().pop(2);
    ASSERT_TRUE(pool.commit());
    EXPECT_EQ(pool.integers().available(), ni - 3);
    ASSERT_TRUE(pool.drop());

    EXPECT_EQ(pool.integers().available(), ni);
}

/// An inner drop reverts only its own allocations, the outer one the rest
TEST_F(buffer_transaction, nested_inner_drop) {
    const int ni = pool.integers().available();

    ASSERT_TRUE(pool.begin());
    pool.integers().pop(1);
    ASSERT_TRUE(pool.begin());
    pool.integers().pop(2);
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().available(), ni - 1);
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().available(), ni);
}

/// Unbalanced and overflowing use is rejected
TEST_F(buffer_transaction, unbalanced_and_overflow) {
    EXPECT_FALSE(pool.commit());
    EXPECT_FALSE(pool.drop());
    for (int i = 0; i < 8; ++i)
        EXPECT_TRUE(pool.begin());
    EXPECT_FALSE(pool.begin());
    EXPECT_EQ(pool.open_transactions(), 8);
}

/// A mixed-precision group allocator restores its mask on drop
TEST_F(buffer_transaction, mixed_group_mask_restored) {
    pool.set_vector_group_size(1, 4);
    const int nin = pool.vectors().available(buffer::allocation_t::input);
    const int nout = pool.vectors().available(buffer::allocation_t::output);

    ASSERT_TRUE(pool.begin());
    pool.vectors().pop(2, buffer::allocation_t::output);
    EXPECT_LT(pool.vectors().available(buffer::allocation_t::input), nin);
    ASSERT_TRUE(pool.drop());

    EXPECT_EQ(pool.vectors().available(buffer::allocation_t::input), nin);
    EXPECT_EQ(pool.vectors().available(buffer::allocation_t::output), nout);
}

// buffer::buffer_pool_t allocation transactions

/// An allocation shortfall reports failure and a drop leaves the pool intact
TEST_F(buffer_allocation, shortfall_then_drop) {
    const int ni = pool.integers().available();

    wants_t too_many(ni + 1);
    ASSERT_TRUE(pool.begin());
    EXPECT_FALSE(pool.allocate(too_many));
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().available(), ni);
}

/// Objects that are gone before drop()/commit() are never touched by them
TEST_F(buffer_allocation, objects_may_die_before_the_transaction_closes) {
    const int ni = pool.integers().available();

    ASSERT_TRUE(pool.begin());
    ASSERT_TRUE(pool.begin());
    {
        wants_t scoped(2);
        ASSERT_TRUE(pool.allocate(scoped));
    }
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().available(), ni);

    ASSERT_TRUE(pool.begin());
    {
        wants_t scoped(2);
        ASSERT_TRUE(pool.allocate(scoped));
    }
    ASSERT_TRUE(pool.commit());
    ASSERT_TRUE(pool.commit());
    EXPECT_EQ(pool.integers().available(), ni - 2);
}

/// A pool-and-stack transaction drops both bookkeeping states together
TEST_F(buffer_allocation, pool_and_stack_drop_together) {
    stack_t stack(pool);
    const int ni = pool.integers().available();

    ASSERT_TRUE(pool.begin());
    ASSERT_TRUE(stack.begin());
    const pointer_t first = stack.new_slot().memory();
    pool.new_buffer_source();
    ASSERT_TRUE(stack.drop());
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().available(), ni);

    // The reclaimed stack slot is handed out again
    EXPECT_EQ(stack.new_slot().memory().offset, first.offset);
}

/// The scope guard drops unless committed
TEST_F(buffer_allocation, scope_guard) {
    stack_t stack(pool);
    const int ni = pool.integers().available();

    {
        transaction_t txn(pool, stack);
        ASSERT_TRUE(txn);
        pool.integers().pop(2);
    }
    EXPECT_EQ(pool.integers().available(), ni);

    {
        transaction_t txn(pool, stack);
        pool.integers().pop(2);
        ASSERT_TRUE(txn.commit());
    }
    EXPECT_EQ(pool.integers().available(), ni - 2);
}
} // namespace rvjit
