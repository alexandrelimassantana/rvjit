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

#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "rvjit/implementation/common/stream/combinator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "stream_mock.hpp"

namespace rvjit {

using stream::on_interleave_t;
using stream::on_nest_t;
using stream::policy_t;
using stream::ravel;
using stream::scheduler_t;
using stream::unravel;

// unravel()/ravel(): inverse of one another

/// ravel() and unravel() are inverses of one another for every flat index
TEST(index_conversions, ravel_unravel_are_inverses) {
    const index_t shape2 {4, 5};
    for (int flat = 0; flat < 4 * 5; ++flat)
        EXPECT_EQ(ravel(unravel(flat, 2, shape2), 2, shape2), flat);
}

// scheduler_t

/// from_params() stores its two arguments as mem_latency/aop_latency
TEST(scheduler, from_params_sets_both_distances) {
    const scheduler_t sched = scheduler_t::from_params(2, 5);
    EXPECT_EQ(sched.mem_latency, 2);
    EXPECT_EQ(sched.aop_latency, 5);
}

// interleaved: one uniform delay shared by every entry

/// Every entry fires at its own position whatever the delay; only `cb` defers
TEST(interleaved, entries_fire_naturally_regardless_of_delay) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    std::vector<int> a_steps, b_steps;
    auto sa = ms.add(
            {5}, [&](const index_t &idx) { a_steps.push_back(idx[0]); });
    auto sb = ms.add(
            {3}, [&](const index_t &idx) { b_steps.push_back(idx[0]); });

    int cb_calls = 0;
    // A grossly oversized delay only affects cb's own deferral -- it
    // never shifts or clamps either entry's own firing
    combinator.interleaved(policy_t::literal(1000),
            [&](const index_t &) { ++cb_calls; }, {sa, sb});

    EXPECT_EQ(cb_calls, 5); // exactly once per combined position
    EXPECT_EQ(a_steps, (std::vector<int> {0, 1, 2, 3, 4}));
    EXPECT_EQ(b_steps, (std::vector<int> {0, 1, 2}));
}

/// `cb` fires once per position for any delay, flushed on the last step
TEST(interleaved, cb_fires_exactly_once_per_position_for_any_delay) {
    common::stream::combinator_t combinator;
    for (const int delay : {0, 1, 4, 1000}) {
        mock_streams_t ms;
        auto sa = ms.add({4});
        auto sb = ms.add({4});
        std::vector<index_t> cb_seen;
        combinator.interleaved(policy_t::literal(delay),
                [&](const index_t &idx) { cb_seen.push_back(idx); }, {sa, sb});
        ASSERT_EQ(cb_seen.size(), 4u) << "delay=" << delay;
        for (int i = 0; i < 4; ++i) {
            const int seen = cb_seen[static_cast<size_t>(i)][0];
            EXPECT_EQ(seen, i) << "delay=" << delay << " i=" << i;
        }
    }
}

/// `cb` sees the per-axis maximum shape across both entries
TEST(interleaved, cb_walks_the_per_axis_max_shape) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    auto sa = ms.add({3, 2});
    auto sb = ms.add({2, 4});
    std::vector<index_t> seen;
    combinator.interleaved(policy_t::zero(),
            [&](const index_t &idx) { seen.push_back(idx); }, {sa, sb});
    ASSERT_EQ(seen.size(), 12u); // 3 x 4
    EXPECT_EQ(seen.back()[0], 2);
    EXPECT_EQ(seen.back()[1], 3);
}

/// Entries due at a step fire in argument order, then `cb` runs once
TEST(interleaved_nary, fires_every_entry_then_invokes_cb_once) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    std::vector<int> order;
    auto sa = ms.add({3}, [&](const index_t &idx) { order.push_back(idx[0]); });
    auto sb = ms.add(
            {3}, [&](const index_t &idx) { order.push_back(100 + idx[0]); });
    auto sc = ms.add(
            {3}, [&](const index_t &idx) { order.push_back(200 + idx[0]); });

    int cb_calls = 0;
    combinator.interleaved(policy_t::zero(),
            [&](const index_t &) { ++cb_calls; }, {sa, sb, sc});

    EXPECT_EQ(cb_calls, 3);
    {
        const std::vector<int> want = {0, 100, 200, 1, 101, 201, 2, 102, 202};
        EXPECT_EQ(order, want);
    }
}

/// Three or more entries fire at their own position whatever the delay
TEST(interleaved_nary, entries_fire_naturally_regardless_of_delay) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    std::vector<int> a_steps, c_steps;
    auto sa = ms.add(
            {5}, [&](const index_t &idx) { a_steps.push_back(idx[0]); });
    auto sb = ms.add({3});
    auto sc = ms.add(
            {2}, [&](const index_t &idx) { c_steps.push_back(idx[0]); });

    combinator.interleaved(
            policy_t::literal(1000), [&](const index_t &) {}, {sa, sb, sc});

    EXPECT_EQ(a_steps, (std::vector<int> {0, 1, 2, 3, 4}));
    EXPECT_EQ(c_steps, (std::vector<int> {0, 1}));
}

/// A stream referenced as `init()` drives its init phase, not production
TEST(interleaved, init_references_drive_the_init_phase) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    int produced = 0, inited = 0;
    auto sa = ms.add({3}, [&](const index_t &) { ++produced; },
            [&](const index_t &) { ++inited; });
    auto sb = ms.add({2}, [&](const index_t &) { ++produced; },
            [&](const index_t &) { ++inited; });
    combinator.interleaved({sa.init(), sb.init()});
    EXPECT_EQ(produced, 0);
    EXPECT_EQ(inited, 3 + 2);
}

/// Fewer than two entries emit nothing
TEST(interleaved, a_single_entry_emits_nothing) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    int calls = 0;
    auto sa = ms.add({3}, [&](const index_t &) { ++calls; });
    combinator.interleaved({sa});
    EXPECT_EQ(calls, 0);
}

/// Entries of different dimensionality emit nothing
TEST(interleaved, mismatched_ndims_emit_nothing) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    int calls = 0;
    auto sa = ms.add({3}, [&](const index_t &) { ++calls; });
    auto sb = ms.add({3, 2}, [&](const index_t &) { ++calls; });
    combinator.interleaved({sa, sb});
    EXPECT_EQ(calls, 0);
}

// interleaved: policy_t resolves delay from the encapsulated scheduler_t

namespace {

/// Compares `interleaved(opt, cb, ...)`'s resolved delay against
/// `interleaved(literal(expected_delay), cb, ...)`
///
/// @note Proves policy_t resolves to that value
void expect_interleave_policy_matches(common::stream::combinator_t &combinator,
        policy_t opt, int expected_delay) {
    std::vector<int> a, b;
    mock_streams_t ms;
    auto cb_log = [](std::vector<int> &v) {
        return [&v](const index_t &idx) { v.push_back(1000 + idx[0]); };
    };
    auto sa0 = ms.add({6}, [&](const index_t &idx) { a.push_back(idx[0]); });
    auto sb0 = ms.add(
            {4}, [&](const index_t &idx) { a.push_back(100 + idx[0]); });
    auto sa1 = ms.add({6}, [&](const index_t &idx) { b.push_back(idx[0]); });
    auto sb1 = ms.add(
            {4}, [&](const index_t &idx) { b.push_back(100 + idx[0]); });

    combinator.interleaved(opt, cb_log(a), {sa0, sb0});
    combinator.interleaved(
            policy_t::literal(expected_delay), cb_log(b), {sa1, sb1});

    EXPECT_EQ(a, b);
}

} // namespace

/// policy_t::memory() resolves interleaved()'s delay to the memory latency
TEST(interleave_policy, memory_resolves_memory_distance) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/2, /*arith=*/9);
    common::stream::combinator_t combinator(sched);
    expect_interleave_policy_matches(
            combinator, policy_t::memory(), sched.mem_latency);
}

/// policy_t::arithmetic() resolves interleaved()'s delay to aop_latency
TEST(interleave_policy, arithmetic_resolves_arithmetic_distance) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/1, /*arith=*/2);
    common::stream::combinator_t combinator(sched);
    expect_interleave_policy_matches(
            combinator, policy_t::arithmetic(), sched.aop_latency);
}

/// policy_t::zero() resolves interleaved()'s delay to 0 for any scheduler
TEST(interleave_policy, zero_ignores_the_scheduler_entirely) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/3, /*arith=*/7);
    common::stream::combinator_t combinator(sched);
    expect_interleave_policy_matches(combinator, policy_t::zero(), 0);
}

// interleaved: cb-less overload, for combining streams whose own per-entry
// callbacks are the entire point

/// The cb-less overload matches the cb-taking one at delay 0 with a no-op cb
TEST(interleaved_no_cb, matches_the_cb_taking_overload_at_zero_delay) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    std::vector<int> a, b;
    auto sa0 = ms.add({6}, [&](const index_t &idx) { a.push_back(idx[0]); });
    auto sb0 = ms.add(
            {4}, [&](const index_t &idx) { a.push_back(100 + idx[0]); });
    auto sa1 = ms.add({6}, [&](const index_t &idx) { b.push_back(idx[0]); });
    auto sb1 = ms.add(
            {4}, [&](const index_t &idx) { b.push_back(100 + idx[0]); });

    combinator.interleaved({sa0, sb0});
    combinator.interleaved(
            policy_t::zero(), [&](const index_t &) {}, {sa1, sb1});

    EXPECT_EQ(a, b);
}

/// The cb-less overload ignores the scheduler, having no cb to defer
TEST(interleaved_no_cb, ignores_the_scheduler_entirely) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/2, /*arith=*/9);
    common::stream::combinator_t combinator(sched);
    common::stream::combinator_t zero_combinator;
    mock_streams_t ms;

    std::vector<int> a, b;
    auto sa0 = ms.add({6}, [&](const index_t &idx) { a.push_back(idx[0]); });
    auto sb0 = ms.add(
            {4}, [&](const index_t &idx) { a.push_back(100 + idx[0]); });
    auto sa1 = ms.add({6}, [&](const index_t &idx) { b.push_back(idx[0]); });
    auto sb1 = ms.add(
            {4}, [&](const index_t &idx) { b.push_back(100 + idx[0]); });

    combinator.interleaved({sa0, sb0});
    zero_combinator.interleaved({sa1, sb1});

    EXPECT_EQ(a, b);
}

} // namespace rvjit
