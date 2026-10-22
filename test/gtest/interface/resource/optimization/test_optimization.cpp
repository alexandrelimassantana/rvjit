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
#include <vector>

#include "gtest/gtest.h"

#include "rvjit/interface/resource/optimization/group.hpp"
#include "rvjit/interface/resource/optimization/search.hpp"

namespace rvjit {

namespace {

/// Optimizable with `n` levels relieving an unknown count of `r` per level
struct level_t : resource_optimizable_t {
    explicit level_t(int n, resource_t r = resource_t::integers)
        : n_(n), r_(r) {}
    int levels() const override { return n_; }
    int level() const override { return l_; }
    bool set_level(int l) override {
        l_ = l;
        return true;
    }
    demand_t relief(int from, int to) const override {
        demand_t d;
        if (from != to) d.unknown = static_cast<unsigned>(r_);
        return d;
    }

private:
    int n_, l_ = 0;
    resource_t r_;
};

/// Optimizable holding `held[level]` integers
struct holder_t : level_t {
    explicit holder_t(int n, int base, int step) : level_t(n) {
        for (int l = 0; l < n; ++l)
            held_.push_back(base - step * l);
    }
    explicit holder_t(std::vector<int> held)
        : level_t(static_cast<int>(held.size())), held_(std::move(held)) {}
    int held() const { return held_[level()]; }
    demand_t demand() const override {
        return demand_t::of(resource_t::integers, held());
    }
    demand_t relief(int from, int to) const override {
        return demand_t::of(resource_t::integers, held_[from] - held_[to]);
    }

private:
    std::vector<int> held_;
};

/// The shortfall of holding `held` integers out of `cap`
shortfall_t over(int held, int cap) {
    return held > cap ? shortfall_t::of(resource_t::integers, held - cap)
                      : shortfall_t {};
}

/// Transactional state counter recording begin/commit/drop
struct counter_t : transactional_t {
    bool begin() override {
        ++open;
        ++begins;
        return true;
    }
    bool commit() override {
        --open;
        ++commits;
        return true;
    }
    bool drop() override {
        --open;
        ++drops;
        return true;
    }
    int open_transactions() const override { return open; }
    int open = 0, begins = 0, commits = 0, drops = 0;
};

} // namespace

// group_t

/// Depth-first relaxes each member fully before the next
TEST(optimization_group, depth_first_order) {
    level_t a(3), b(2);
    optimization::group_t g(optimization::order_t::depth_first);
    g.declare(a, b);
    ASSERT_EQ(g.levels(), 4);

    const int exp[4][2] = {{0, 0}, {1, 0}, {2, 0}, {2, 1}};
    for (int l = 0; l < 4; ++l) {
        ASSERT_TRUE(g.set_level(l));
        EXPECT_EQ(a.level(), exp[l][0]) << l;
        EXPECT_EQ(b.level(), exp[l][1]) << l;
        EXPECT_EQ(g.level(), l);
    }
}

/// Level-first relaxes every member one level before any goes further
TEST(optimization_group, level_first_order) {
    level_t a(3), b(3), c(3);
    optimization::group_t g(optimization::order_t::level_first);
    g.declare(a, b, c);
    ASSERT_EQ(g.levels(), 7);

    const int exp[7][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1},
            {2, 1, 1}, {2, 2, 1}, {2, 2, 2}};
    for (int l = 0; l < 7; ++l) {
        ASSERT_TRUE(g.set_level(l));
        EXPECT_EQ(a.level(), exp[l][0]) << l;
        EXPECT_EQ(b.level(), exp[l][1]) << l;
        EXPECT_EQ(c.level(), exp[l][2]) << l;
        EXPECT_EQ(g.level(), l);
    }
}

/// Members with fewer levels drop out of the level-first walk
TEST(optimization_group, level_first_uneven) {
    level_t a(2), b(3);
    optimization::group_t g(optimization::order_t::level_first);
    g.declare(a, b);
    ASSERT_EQ(g.levels(), 4);

    const int exp[4][2] = {{0, 0}, {1, 0}, {1, 1}, {1, 2}};
    for (int l = 0; l < 4; ++l) {
        ASSERT_TRUE(g.set_level(l));
        EXPECT_EQ(a.level(), exp[l][0]) << l;
        EXPECT_EQ(b.level(), exp[l][1]) << l;
    }
}

/// Setting a lower level after a higher one is absolute
TEST(optimization_group, set_level_is_absolute) {
    level_t a(3), b(3);
    optimization::group_t g;
    g.declare(a, b);
    ASSERT_TRUE(g.set_level(4));
    ASSERT_TRUE(g.set_level(1));
    EXPECT_EQ(a.level(), 1);
    EXPECT_EQ(b.level(), 0);
    EXPECT_FALSE(g.set_level(g.levels()));
}

/// A group relieves whatever its members relieve
TEST(optimization_group, relieves_union) {
    level_t a(2, resource_t::integers), b(2, resource_t::stack);
    optimization::group_t g;
    g.declare(a, b);
    EXPECT_TRUE(g.relieves().intersects(shortfall_t::of(resource_t::integers)));
    EXPECT_TRUE(g.relieves().intersects(shortfall_t::of(resource_t::stack)));
    EXPECT_FALSE(g.relieves().intersects(shortfall_t::of(resource_t::vectors)));
}

/// A group's relief sums its members' at their shares of each level
TEST(optimization_group, relief_sums_the_members) {
    holder_t a(3, 10, 2), b(2, 5, 3);
    optimization::group_t g;
    g.declare(a, b);
    EXPECT_EQ(g.relief(0, 2)[resource_t::integers], 4);
    EXPECT_EQ(g.relief(0, 3)[resource_t::integers], 7);
    EXPECT_EQ(g.relief(3, 1)[resource_t::integers], -5);
    EXPECT_EQ(g.demand()[resource_t::integers], 15);
}

/// relieves() follows the level: nothing past the last or a no-gain level
TEST(optimization_group, relieves_is_dynamic) {
    holder_t h({6, 4, 4});
    EXPECT_FALSE(h.relieves().ok());
    ASSERT_TRUE(h.set_level(1));
    EXPECT_TRUE(h.relieves().ok());
}

// search_t

/// Candidates are visited with the first axis slowest
TEST(optimization_search, visit_order) {
    level_t a(2), b(3);
    counter_t txn;
    optimization::search_t s;
    s.guard(txn);
    s.axis(a, b);

    std::vector<std::pair<int, int>> seen;
    {
        const auto build = [&] {
            seen.emplace_back(a.level(), b.level());
            return shortfall_t::of(resource_t::integers);
        };
        EXPECT_FALSE(s.run(build));
    }
    const std::vector<std::pair<int, int>> exp
            = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
    EXPECT_EQ(seen, exp);
    EXPECT_EQ(txn.drops, 6);
    EXPECT_EQ(txn.commits, 0);
    EXPECT_EQ(txn.open, 0);
}

/// The first fitting candidate wins and is committed
TEST(optimization_search, first_fit_commits) {
    level_t a(2), b(3);
    counter_t txn;
    optimization::search_t s;
    s.guard(txn);
    s.axis(a, b);

    {
        const auto build = [&] {
            return a.level() == 1 && b.level() == 1
                    ? shortfall_t {}
                    : shortfall_t::of(resource_t::integers);
        };
        EXPECT_TRUE(s.run(build));
    }
    EXPECT_EQ(a.level(), 1);
    EXPECT_EQ(b.level(), 1);
    EXPECT_EQ(txn.commits, 1);
    EXPECT_EQ(txn.drops, 4);
    EXPECT_EQ(txn.open, 0);
}

/// Every guard sees the same begin/commit/drop sequence
TEST(optimization_search, guards_in_lockstep) {
    level_t a(2);
    counter_t t1, t2;
    optimization::search_t s;
    s.guard(t1, t2);
    s.axis(a);
    {
        const auto build = [&] {
            return a.level() == 1 ? shortfall_t {}
                                  : shortfall_t::of(resource_t::integers);
        };
        EXPECT_TRUE(s.run(build));
    }
    EXPECT_EQ(t1.begins, 2);
    EXPECT_EQ(t2.begins, 2);
    EXPECT_EQ(t1.commits, 1);
    EXPECT_EQ(t2.commits, 1);
    EXPECT_EQ(t1.drops, 1);
    EXPECT_EQ(t2.drops, 1);
}

/// A shortfall the faster axes cannot relieve skips their remaining levels
TEST(optimization_search, prunes_unrelieving_axes) {
    level_t slow(2, resource_t::vectors); // relieves vectors
    level_t fast(4, resource_t::integers); // relieves integers only
    counter_t txn;
    optimization::search_t s;
    s.guard(txn);
    s.axis(slow, fast);

    int calls = 0;
    {
        const auto build = [&] {
            ++calls;
            return slow.level() == 1 ? shortfall_t {}
                                     : shortfall_t::of(resource_t::vectors);
        };
        EXPECT_TRUE(s.run(build));
    }
    // (0,0) fails on vectors: `fast` cannot help, so (0,1..3) are skipped
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(slow.level(), 1);
    EXPECT_EQ(fast.level(), 0);
}

/// A relieving shortfall still walks the faster axis
TEST(optimization_search, relieving_axis_is_walked) {
    level_t slow(2, resource_t::vectors);
    level_t fast(3, resource_t::integers);
    counter_t txn;
    optimization::search_t s;
    s.guard(txn);
    s.axis(slow, fast);

    {
        const auto build = [&] {
            return fast.level() == 2 ? shortfall_t {}
                                     : shortfall_t::of(resource_t::integers);
        };
        EXPECT_TRUE(s.run(build));
    }
    EXPECT_EQ(slow.level(), 0);
    EXPECT_EQ(fast.level(), 2);
}

/// No axes means a single candidate
TEST(optimization_search, no_axes) {
    counter_t txn;
    optimization::search_t s;
    s.guard(txn);
    EXPECT_TRUE(s.run([] { return shortfall_t {}; }));
    EXPECT_EQ(txn.commits, 1);
    EXPECT_FALSE(s.run([] { return shortfall_t::of(resource_t::stack); }));
}

// search_t ruling candidates out by registers freed

/// A failure with a known deficit skips the levels that cannot free it
TEST(optimization_search, skips_levels_that_cannot_free_the_deficit) {
    holder_t ring(10, 20, 2);
    optimization::search_t s;
    s.axis(ring);
    EXPECT_TRUE(s.run([&] { return over(ring.held(), 8); }));
    EXPECT_EQ(ring.level(), 6);
    // The level-0 failure and the first level freeing 12, against seven
    EXPECT_EQ(s.stats().builds, 2);
}

/// Without a known deficit the search builds every level that frees any
TEST(optimization_search, single_steps_without_a_known_deficit) {
    holder_t h(6, 12, 2);
    optimization::search_t s;
    s.axis(h);
    {
        const auto build = [&] {
            return h.level() >= 3 ? shortfall_t {}
                                  : shortfall_t::of(resource_t::integers);
        };
        EXPECT_TRUE(s.run(build));
    }
    EXPECT_EQ(h.level(), 3);
    EXPECT_EQ(s.stats().builds, 4);
}

/// A level freeing nothing is never built, even on a non-monotone axis
TEST(optimization_search, skips_levels_freeing_nothing) {
    holder_t h({10, 10, 12, 7, 9, 6});
    optimization::search_t s;
    s.axis(h);
    int builds = 0;
    {
        const auto build = [&] {
            ++builds;
            return over(h.held(), 8);
        };
        EXPECT_TRUE(s.run(build));
    }
    EXPECT_EQ(h.level(), 3);
    EXPECT_EQ(builds, 2);
}

/// With exact reports the pruned search lands where the exhaustive one does
TEST(optimization_search, pruning_matches_the_exhaustive_search) {
    for (int cap = 0; cap <= 20; ++cap) {
        const auto run = [&](bool verify) {
            holder_t slow({16, 14, 15, 9}), fast({6, 6, 3, 1});
            optimization::search_t s;
            s.axis(slow, fast).verify(verify);
            const bool ok = s.run(
                    [&] { return over(slow.held() + fast.held(), cap); });
            EXPECT_EQ(s.stats().mispruned, 0) << "capacity " << cap;
            return ok ? slow.level() * 10 + fast.level() : -1;
        };
        EXPECT_EQ(run(false), run(true)) << "capacity " << cap;
    }
}

/// The slow axis advances only once the faster one cannot free enough
TEST(optimization_search, a_slow_step_never_precedes_a_faster_axis) {
    holder_t slow(8, 16, 2);
    level_t fast(2);
    optimization::search_t s;
    s.axis(slow, fast);
    EXPECT_TRUE(s.run([&] { return over(slow.held(), 6); }));
    EXPECT_EQ(fast.level(), 0);
    EXPECT_EQ(slow.level(), 5);
}

/// A deficit no combination of levels can free fails after one build
TEST(optimization_search, fails_fast_when_nothing_frees_enough) {
    holder_t a(4, 10, 1), b(3, 10, 2);
    optimization::search_t s;
    s.axis(a, b);
    EXPECT_FALSE(s.run([&] { return over(a.held() + b.held(), 5); }));
    EXPECT_EQ(s.stats().builds, 1);
}

/// Every recorded failure must be covered, not only the last one
TEST(optimization_search, every_failure_rules_candidates_out) {
    holder_t ints(4, 8, 2);
    level_t vecs(3, resource_t::vectors);
    optimization::search_t s;
    s.axis(vecs, ints);
    std::vector<std::pair<int, int>> seen;
    {
        const auto build = [&] {
            seen.emplace_back(vecs.level(), ints.level());
            if (vecs.level() < 2) return shortfall_t::of(resource_t::vectors);
            return over(ints.held(), 3);
        };
        EXPECT_TRUE(s.run(build));
    }
    // (0, 1..3) cannot free vectors; (2, 0..2) cannot free the 5 integers
    const std::vector<std::pair<int, int>> exp
            = {{0, 0}, {1, 0}, {2, 0}, {2, 3}};
    EXPECT_EQ(seen, exp);
}

} // namespace rvjit
