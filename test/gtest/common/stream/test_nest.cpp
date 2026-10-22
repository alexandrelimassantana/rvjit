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
#include "rvjit/interface/stream/buffer_map.hpp"
#include "stream_mock.hpp"

namespace rvjit {

using stream::on_interleave_t;
using stream::on_nest_t;
using stream::policy_t;
using stream::ravel;
using stream::scheduler_t;
using stream::unravel;

// nested: shared axes, own axes and callback arity

/// `cb` receives each stream's full per-axis idx, the shared axes first
TEST(nested_shape, cb_receives_full_indices_of_both_streams) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    auto s1 = ms.add({2, 3}); // {K, M}
    auto s2 = ms.add({2, 4}); // {K, N}
    int calls = 0;
    combinator.nested(1, policy_t::zero(), s1, s2,
            [&](const index_t &i1, const index_t &i2) {
        EXPECT_EQ(i1[0], i2[0]);
        EXPECT_LT(i1[1], 3);
        EXPECT_LT(i2[1], 4);
        ++calls;
    });
    EXPECT_EQ(calls, 2 * 3 * 4);
}

/// Zero shared axes cartesian-multiply every axis of both streams
TEST(nested_shape, degenerate_ns_zero_covers_every_axis_combination) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    auto s1 = ms.add({2, 3});
    auto s2 = ms.add({2, 4});
    int calls = 0;
    combinator.nested(0, policy_t::zero(), s1, s2,
            [&](const index_t &, const index_t &) { ++calls; });
    EXPECT_EQ(calls, 2 * 3 * 2 * 4);
}

/// A shared axis of size 0 makes the whole nest a no-op
TEST(nested_shape, shared_axis_of_size_zero_is_a_noop) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    int calls = 0;
    auto s1 = ms.add({0, 3}, [&](const index_t &) { ++calls; });
    auto s2 = ms.add({0, 4}, [&](const index_t &) { ++calls; });
    combinator.nested(1, policy_t::zero(), s1, s2,
            [&](const index_t &, const index_t &) { ++calls; });
    EXPECT_EQ(calls, 0);
}

/// Shared axes of different sizes emit nothing
TEST(nested_shape, mismatched_shared_axes_emit_nothing) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    int calls = 0;
    auto s1 = ms.add({2, 3}, [&](const index_t &) { ++calls; });
    auto s2 = ms.add({3, 4}, [&](const index_t &) { ++calls; });
    combinator.nested(1, policy_t::zero(), s1, s2,
            [&](const index_t &, const index_t &) { ++calls; });
    EXPECT_EQ(calls, 0);
}

// nested: delay semantics (s2's production timing)

/// delay == 0 produces every s2 position exactly once, at its first read
TEST(nested_delay, zero_produces_s2_once_each_shared_position) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    const int K = 3, M = 4, N = 5;
    std::vector<int> s2_produced; // records k*100+n
    std::vector<int> s1_produced; // records k*100+m
    std::set<std::tuple<int, int, int>> cb_seen;
    bool k_mismatch = false;

    auto s1 = ms.add({K, M}, [&](const index_t &idx) {
        s1_produced.push_back(idx[0] * 100 + idx[1]);
    });
    auto s2 = ms.add({K, N}, [&](const index_t &idx) {
        s2_produced.push_back(idx[0] * 100 + idx[1]);
    });

    combinator.nested_1d(policy_t::zero(), s1, s2,
            [&](const index_t &i1, const index_t &i2) {
        if (i1[0] != i2[0]) k_mismatch = true;
        cb_seen.insert({i1[0], i1[1], i2[1]});
    });

    EXPECT_FALSE(k_mismatch);
    EXPECT_EQ(cb_seen.size(), static_cast<size_t>(K * M * N));
    ASSERT_EQ(s1_produced.size(), static_cast<size_t>(K * M));
    ASSERT_EQ(s2_produced.size(), static_cast<size_t>(K * N));

    // s2 is never re-produced: every recorded (k,n) is unique
    std::set<int> s2_unique(s2_produced.begin(), s2_produced.end());
    EXPECT_EQ(s2_unique.size(), s2_produced.size());
}

/// A positive delay produces s1/s2 once each and only defers cb
TEST(nested_delay, greater_than_zero_leaves_production_unaffected) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    const int K = 3, M = 4, N = 5;
    std::vector<int> s1_produced, s2_produced;
    std::vector<std::tuple<int, int, int>> cb_calls; // (k1, k2, n) per call
    auto s1 = ms.add({K, M}, [&](const index_t &idx) {
        s1_produced.push_back(idx[0] * 100 + idx[1]);
    });
    auto s2 = ms.add({K, N}, [&](const index_t &idx) {
        s2_produced.push_back(idx[0] * 100 + idx[1]);
    });

    combinator.nested_1d(policy_t::literal(2), s1, s2,
            [&](const index_t &i1, const index_t &i2) {
        cb_calls.emplace_back(i1[0], i2[0], i2[1]);
    });

    std::set<int> s1_unique(s1_produced.begin(), s1_produced.end());
    std::set<int> s2_unique(s2_produced.begin(), s2_produced.end());
    EXPECT_EQ(s1_unique.size(), static_cast<size_t>(K * M));
    EXPECT_EQ(s2_unique.size(), static_cast<size_t>(K * N));
    EXPECT_EQ(s1_produced.size(), s1_unique.size());
    EXPECT_EQ(s2_produced.size(), s2_unique.size());

    // cb still fires exactly once per (k, m, n) triple overall
    EXPECT_EQ(cb_calls.size(), static_cast<size_t>(K * M * N));
    // a delay of 2, smaller than N (5), never crosses a shared position --
    // every call's own two idx still agree on k
    for (const auto &c : cb_calls)
        EXPECT_EQ(std::get<0>(c), std::get<1>(c));
}

/// A delay of one row still fires cb once per (k, n) across row boundaries
TEST(nested_delay, can_cross_a_shared_position_boundary) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    const int K = 3, M = 1, N = 4; // total = M*N = 4 per row
    const int total = M * N;
    std::vector<int> cb_seen; // records k*100+n
    auto s1 = ms.add({K, M});
    auto s2 = ms.add({K, N});

    combinator.nested_1d(policy_t::literal(total), s1, s2,
            [&](const index_t &i1, const index_t &i2) {
        EXPECT_EQ(i1[0], i2[0]); // both idx always share the same row
        cb_seen.push_back(i1[0] * 100 + i2[1]);
    });

    ASSERT_EQ(cb_seen.size(), static_cast<size_t>(K * M * N));
    std::set<int> cb_unique(cb_seen.begin(), cb_seen.end());
    EXPECT_EQ(cb_unique.size(), cb_seen.size());
}

/// A delay past the whole nest fires every cb once, bunched in the last step
TEST(nested_delay, delay_exceeding_the_whole_nest_bunches_at_the_end) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    const int K = 2, M = 3, N = 7;
    int cb_calls = 0;
    auto s1 = ms.add({K, M});
    auto s2 = ms.add({K, N});

    // delay grossly exceeds K*M*N (42); every cb call must still happen
    // exactly once
    combinator.nested_1d(policy_t::literal(1000), s1, s2,
            [&](const index_t &, const index_t &) { ++cb_calls; });

    EXPECT_EQ(cb_calls, K * M * N);
}

/// Production and cb order match a cell-by-cell reference (k outer, mn inner)
TEST(nested_ordering, matches_naive_per_shared_position_reference) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    // delay (7) exceeds one row's own total (M*N=6)
    const int K = 3, M = 2, N = 3, delay = 7;
    const int total = M * N;
    const int grand_total = K * total;
    std::vector<int> actual, expect;

    auto s1 = ms.add({K, M}, [&](const index_t &idx) {
        actual.push_back(1000 + idx[0] * 100 + idx[1]);
    });
    auto s2 = ms.add({K, N}, [&](const index_t &idx) {
        actual.push_back(2000 + idx[0] * 100 + idx[1]);
    });
    combinator.nested_1d(policy_t::literal(delay), s1, s2,
            [&](const index_t &i1, const index_t &i2) {
        actual.push_back(3000 + i1[0] * 100 + i1[1] * 10 + i2[1]);
    });

    auto push_cb = [&](int p) {
        const int kf = p / total, mn2 = p % total;
        expect.push_back(3000 + kf * 100 + (mn2 / N) * 10 + mn2 % N);
    };

    for (int k = 0; k < K; ++k) {
        for (int mn = 0; mn < total; ++mn) {
            const int m = mn / N, n = mn % N;
            if (m == 0) expect.push_back(2000 + k * 100 + n);
            if (n == 0) expect.push_back(1000 + k * 100 + m);

            // cb is deferred by `delay`, globally across every row; the
            // very last position of the whole nest flushes every
            // remaining not-yet-fired position
            const int g = k * total + mn;
            if (g < grand_total - 1) {
                const int p = g - delay;
                if (p >= 0) push_cb(p);
            } else {
                for (int p = std::max(0, g - delay); p < grand_total; ++p)
                    push_cb(p);
            }
        }
    }

    EXPECT_EQ(actual, expect);
}

// nested: policy_t resolves delay from the encapsulated scheduler_t

namespace {

/// Compares `nested_1d(opt, ...)`'s resolved delay against
/// `nested_1d(literal(expected_delay), ...)`
///
/// @note Proves policy_t resolves to that value
void expect_nest_policy_matches(common::stream::combinator_t &combinator,
        policy_t opt, int expected_delay) {
    std::vector<int> a, b;
    mock_streams_t ms;
    auto s1a = ms.add({2, 3});
    auto s2a = ms.add({2, 6});
    auto s1b = ms.add({2, 3});
    auto s2b = ms.add({2, 6});

    combinator.nested_1d(
            opt, s1a, s2a, [&](const index_t &i1, const index_t &i2) {
        a.push_back(i1[0] * 1000 + i1[1] * 100 + i2[1]);
    });
    combinator.nested_1d(policy_t::literal(expected_delay), s1b, s2b,
            [&](const index_t &i1, const index_t &i2) {
        b.push_back(i1[0] * 1000 + i1[1] * 100 + i2[1]);
    });

    EXPECT_EQ(a, b);
}

} // namespace

/// policy_t::memory() resolves nested()'s delay to the scheduler's mem_latency
TEST(nest_policy, memory_resolves_memory_distance) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/2, /*arith=*/9);
    common::stream::combinator_t combinator(sched);
    expect_nest_policy_matches(
            combinator, policy_t::memory(), sched.mem_latency);
}

/// policy_t::arithmetic() resolves nested()'s delay to aop_latency
TEST(nest_policy, arithmetic_resolves_arithmetic_distance) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/1, /*arith=*/2);
    common::stream::combinator_t combinator(sched);
    expect_nest_policy_matches(
            combinator, policy_t::arithmetic(), sched.aop_latency);
}

/// policy_t::zero() resolves nested()'s delay to 0 for any scheduler
TEST(nest_policy, zero_ignores_the_scheduler_entirely) {
    const scheduler_t sched
            = scheduler_t::from_params(/*memory=*/3, /*arith=*/7);
    common::stream::combinator_t combinator(sched);
    expect_nest_policy_matches(combinator, policy_t::zero(), 0);
}

/// nested_1d() matches nested(1, ...) exactly
TEST(nested_1d, matches_nested_ns_1) {
    common::stream::combinator_t combinator;
    mock_streams_t ms;
    auto s1a = ms.add({2, 3});
    auto s2a = ms.add({2, 4});
    auto s1b = ms.add({2, 3});
    auto s2b = ms.add({2, 4});
    std::vector<int> a, b;

    combinator.nested_1d(policy_t::literal(1), s1a, s2a,
            [&](const index_t &i1, const index_t &i2) {
        a.push_back(i1[0] * 1000 + i1[1] * 100 + i2[1]);
    });
    combinator.nested(1, policy_t::literal(1), s1b, s2b,
            [&](const index_t &i1, const index_t &i2) {
        b.push_back(i1[0] * 1000 + i1[1] * 100 + i2[1]);
    });

    EXPECT_EQ(a, b);
}

namespace {

/// A host-side stream over a register ring, which emits nothing
struct ring_stream_t : public stream::stream_impl_t {
    ring_stream_t(const index_t &shape, const buffer_map_t &ring)
        : shape_(shape), ring_(ring) {}

    int ndims() const override { return NDIMS; }
    int size(int d) const override { return shape_[d]; }
    bool is_valid() const override { return true; }
    name_t get(const index_t &idx) const override { return ring_(idx); }

protected:
    void produce_(const index_t &) override {}

private:
    index_t shape_;
    buffer_map_t ring_;
};

buffer::buffer_t ring_regs(int n) {
    name_t ids[32];
    for (int i = 0; i < n; ++i)
        ids[i] = 1 + i;
    return buffer::buffer_t(ids, n);
}

} // namespace

/// The tolerated delay is the tighter of the two operand rings
TEST(stream_nest_max_delay, follows_the_operand_rings) {
    static constexpr int K = 4, M = 2, N = 3;
    common::stream::combinator_t combinator;

    const auto bound = [&](int depth, int b_depth) {
        // A: {k, m} on a flat ring in production order; B: {k, n}
        ring_stream_t a(
                {K, M}, buffer_map_t(ring_regs(depth), {K, M}).wrapped(depth));
        ring_stream_t b(
                {K, N}, buffer_map_t(ring_regs(b_depth * N), {b_depth, N}));
        return combinator.max_nested_delay(1, a.ref(), b.ref());
    };

    // Independent depths: A shallower than B, equal, and deeper
    EXPECT_EQ(bound(1, 2), 0);
    EXPECT_EQ(bound(2, 2), N);
    EXPECT_EQ(bound(3, 4), 2 * N);
    EXPECT_EQ(bound(2, 4), N);
    EXPECT_EQ(bound(1, 4), 0);
    EXPECT_EQ(bound(3, 2), 2 * N);
    // A B ring of one K row binds whatever A's depth
    EXPECT_EQ(bound(3, 1), N - 1);
    EXPECT_EQ(bound(2, 1), N - 1);
}

} // namespace rvjit
