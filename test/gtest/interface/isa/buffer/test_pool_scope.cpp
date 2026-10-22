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

#include <utility>

#include "isa_any_fixture.hpp"
#include "gtest/gtest.h"

namespace rvjit {

namespace {

buffer::buffer_t v0_only() {
    const name_t v0 = 0;
    return buffer::buffer_t(&v0, 1);
}

buffer::buffer_t one(name_t r) {
    return buffer::buffer_t(&r, 1);
}

} // namespace

struct buffer_scope : isa_any_pool_test_t {};
struct buffer_layout : isa_any_pool_test_t {};
struct buffer_reserved : isa_any_pool_test_t {};
struct buffer_take : isa_any_pool_test_t {};

// buffer_pool_t::scope()

/// A sibling scope draws again the registers its predecessor freed, in order
TEST_F(buffer_scope, siblings_reuse_the_same_registers) {
    pool.set_vector_group_size(1);
    pool.integers().pop(2);
    const int ni = pool.integers().available();
    const int nv = pool.vectors().available();

    buffer::buffer_t first_ints, first_vecs;
    pool.scope([&] {
        first_ints = pool.integers().pop(3);
        first_vecs = pool.vectors().pop(4);
        EXPECT_EQ(pool.integers().available(), ni - 3);
        return shortfall_t {};
    });
    pool.scope([&] {
        EXPECT_EQ(pool.integers().available(), ni);
        EXPECT_EQ(pool.vectors().available(), nv);
        EXPECT_EQ(pool.integers().pop(3), first_ints);
        EXPECT_EQ(pool.vectors().pop(4), first_vecs);
        return shortfall_t {};
    });
}

/// Allocations after a child scope draw again the registers the child used
TEST_F(buffer_scope, enclosing_allocations_reuse_closed_children) {
    const int ni = pool.integers().available();

    buffer::buffer_t inner;
    pool.scope([&] {
        inner = pool.integers().pop(3);
        return shortfall_t {};
    });
    EXPECT_EQ(pool.integers().available(), ni);
    EXPECT_EQ(pool.integers().pop(3), inner);
}

/// A nested scope frees what it drew, keeping its parent's registers
TEST_F(buffer_scope, nested_scopes_keep_the_parent_registers) {
    const int ni = pool.integers().available();

    pool.scope([&] {
        const name_t outer = pool.integers().pop();
        pool.scope([&] {
            EXPECT_NE(pool.integers().pop(), outer);
            return shortfall_t {};
        });
        EXPECT_EQ(pool.integers().available(), ni - 1);
        return shortfall_t {};
    });
}

/// The scope returns the shortfall its body reported
TEST_F(buffer_scope, returns_the_body_shortfall) {
    const int ni = pool.integers().available();
    const shortfall_t s
            = pool.scope([&] { return pool.integers().take(ni + 1).miss(); });
    EXPECT_TRUE(s.intersects(shortfall_t::of(resource_t::integers)));
    EXPECT_EQ(pool.integers().available(), ni);
}

/// Given registers are drawn first, and only inside their scope
TEST_F(buffer_scope, given_registers_are_drawn_first_inside_the_scope) {
    pool.integers().pop(3);
    const name_t owned = pool.integers().pop();
    const int ni = pool.integers().available();

    pool.scope(one(owned), [&] {
        EXPECT_EQ(pool.integers().available(), ni + 1);
        EXPECT_EQ(pool.integers().pop(), owned);
        return shortfall_t {};
    });
    EXPECT_EQ(pool.integers().available(), ni);
    pool.scope(one(owned), [&] {
        EXPECT_EQ(pool.integers().pop(), owned);
        return shortfall_t {};
    });
}

/// A given register stays given to the scope after its children close
TEST_F(buffer_scope, given_register_stays_given_after_a_child_closes) {
    const name_t owned = pool.integers().pop();

    pool.scope(one(owned), [&] {
        pool.scope([&] {
            EXPECT_EQ(pool.integers().pop(), owned);
            return shortfall_t {};
        });
        pool.scope([&] {
            EXPECT_EQ(pool.integers().pop(), owned);
            return shortfall_t {};
        });
        EXPECT_EQ(pool.integers().pop(), owned);
        return shortfall_t {};
    });
}

/// A register outside the candidate list may be given as well
TEST_F(buffer_scope, non_candidate_registers_may_be_given) {
    const name_t outside = 1; // not in the any-ISA list
    const int ni = pool.integers().available();

    pool.scope(one(outside), [&] {
        EXPECT_EQ(pool.integers().available(), ni + 1);
        EXPECT_EQ(pool.integers().pop(), outside);
        return shortfall_t {};
    });
    EXPECT_EQ(pool.integers().available(), ni);
}

/// `used()` keeps what closed scopes drew, a dropped transaction forgets it
TEST_F(buffer_scope, used_survives_scopes_but_not_a_dropped_transaction) {
    const name_t kept = pool.integers().pop();

    ASSERT_TRUE(pool.begin());
    buffer::buffer_t inner;
    pool.scope([&] {
        inner = pool.integers().pop(2);
        return shortfall_t {};
    });
    EXPECT_EQ(pool.integers().used(), one(kept) | inner);
    ASSERT_TRUE(pool.drop());
    EXPECT_EQ(pool.integers().used(), one(kept));
}

// allocator_t::take()

/// take() names the resource class of the allocator that fell short
TEST_F(buffer_take, reports_the_lacking_resource) {
    pool.set_vector_group_size(1);

    const auto ok = pool.integers().take(3);
    ASSERT_TRUE(static_cast<bool>(ok));
    EXPECT_EQ(ok->size(), 3);

    const shortfall_t lack_int = shortfall_t::of(resource_t::integers);
    const shortfall_t lack_flt = shortfall_t::of(resource_t::floats);
    const shortfall_t lack_vec = shortfall_t::of(resource_t::vectors);

    const auto no_ints = pool.integers().take(1000);
    const auto no_floats = pool.floats().take(1000);
    const auto no_vectors = pool.vectors().take(1000);

    EXPECT_FALSE(static_cast<bool>(no_ints));
    EXPECT_TRUE(no_ints.miss().intersects(lack_int));
    EXPECT_TRUE(no_floats.miss().intersects(lack_flt));
    EXPECT_TRUE(no_vectors.miss().intersects(lack_vec));
}

/// A failed take() allocates nothing, and taking zero is a success
TEST_F(buffer_take, failure_leaves_the_pool_untouched) {
    const int n = pool.integers().available();
    EXPECT_FALSE(static_cast<bool>(pool.integers().take(n + 1)));
    EXPECT_EQ(pool.integers().available(), n);
    EXPECT_TRUE(static_cast<bool>(pool.integers().take(0)));
}

// buffer_pool_t::configure_vectors()

/// A layout groups the vectors, and is refused inside a transaction
TEST_F(buffer_layout, configure_vectors) {

    buffer::vector_layout_t layout;
    layout.in = 2;
    layout.out = 4;
    layout.input_only = v0_only();
    ASSERT_TRUE(pool.configure_vectors(layout));
    EXPECT_EQ(pool.vectors().pop(buffer::allocation_t::input), 0);
    EXPECT_GE(pool.vectors().pop(buffer::allocation_t::output), 4);

    ASSERT_TRUE(pool.begin());
    EXPECT_FALSE(pool.configure_vectors(layout));
    ASSERT_TRUE(pool.drop());
}

// Input-only vector registers

/// Outputs never draw an input-only register, in either view mode
TEST_F(buffer_reserved, outputs_skip_the_reserved_group) {
    for (const auto &gs : {std::make_pair(4, 4), std::make_pair(2, 4)}) {
        isa_any_pool_t fresh(arch);
        fresh.set_vector_group_size(gs.first, gs.second, v0_only());
        auto &v = fresh.vectors();
        // Drain every output group: none holds v0
        while (v.available(buffer::allocation_t::output) > 0) {
            const name_t g = v.pop(buffer::allocation_t::output);
            EXPECT_TRUE(g >= gs.second) << "group at v" << int(g);
        }
    }
}

/// Scratch draws the output groups first and the reserved group last
TEST_F(buffer_reserved, scratch_takes_the_reserved_group_last) {
    for (const auto &gs : {std::make_pair(4, 4), std::make_pair(2, 4)}) {
        isa_any_pool_t fresh(arch);
        fresh.set_vector_group_size(gs.first, gs.second, v0_only());
        auto &v = fresh.vectors();
        const int outs = v.available(buffer::allocation_t::output);
        EXPECT_EQ(v.available(buffer::allocation_t::scratch), outs + 1);
        for (int i = 0; i < outs; ++i)
            EXPECT_GE(v.pop(buffer::allocation_t::scratch), gs.second);
        EXPECT_EQ(v.pop(buffer::allocation_t::scratch), 0);
        EXPECT_EQ(v.available(buffer::allocation_t::scratch), 0);
    }
}

/// Inputs draw the reserved group first
TEST_F(buffer_reserved, inputs_start_at_v0) {
    for (const auto &gs : {std::make_pair(4, 4), std::make_pair(2, 4)}) {
        isa_any_pool_t fresh(arch);
        fresh.set_vector_group_size(gs.first, gs.second, v0_only());
        EXPECT_EQ(fresh.vectors().pop(buffer::allocation_t::input), 0);
    }
}

/// The reserved group stays out of reach of output budget checks
TEST_F(buffer_reserved, may_reserve_excludes_the_group) {
    pool.set_vector_group_size(4, 4, v0_only());
    auto &v = pool.vectors();
    const int groups = 32 / 4;
    EXPECT_TRUE(v.may_reserve(groups - 1, 0));
    EXPECT_FALSE(v.may_reserve(groups, 0));
    EXPECT_TRUE(v.may_reserve(groups - 2, 2));
    EXPECT_FALSE(v.may_reserve(groups - 1, 2));
}

} // namespace rvjit
