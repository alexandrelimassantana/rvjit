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

#include "isa_any_registers.hpp"
#include "rvjit/interface/loop/schedule.hpp"

namespace rvjit {

using namespace isa_any;

using schedule_mode_t = loop::schedule_t::mode_t;

namespace {
std::vector<int> factors_of(const loop::schedule_t &s) {
    std::vector<int> v;
    for (int i = 0; i < s.factors.size(); ++i)
        v.push_back(s.factors[i]);
    return v;
}

loop::schedule_t reduce(const size_parameter_t &size, int max_ur, int g = 1) {
    return loop::schedule_t::of(loop::plan_t::reduce(size, max_ur, g));
}

loop::schedule_t parallel(const size_parameter_t &size, int max_ur, int g = 1) {
    return loop::schedule_t::of(loop::plan_t::data_parallel(size, max_ur, g));
}

size_parameter_t runtime() {
    return size_parameter_t::runtime(r10);
}

size_parameter_t less_eq(int bound) {
    return size_parameter_t::less_eq(r10, bound);
}
} // namespace

/// An unbounded size keeps a main stage repeating at the unroll factor
TEST(schedule_t, unbounded_reduce_is_pow2_breakdown_of_max_ur) {
    const auto s = reduce(runtime(), 8);
    EXPECT_TRUE(s.mode == schedule_mode_t::unroll);
    EXPECT_EQ(factors_of(s), (std::vector<int> {8, 4, 2, 1}));
    EXPECT_EQ(s.bound, 0);

    {
        const std::vector<int> want = {6, 4, 2, 1};
        EXPECT_EQ(factors_of(reduce(runtime(), 6)), want);
    }
    EXPECT_EQ(factors_of(reduce(runtime(), 1)), (std::vector<int> {1}));
}

/// An unbounded data-parallel size repeats the unroll, then switches the tail
TEST(schedule_t, unbounded_data_parallel_granularity1_switches_the_tail) {
    const auto s = parallel(runtime(), 5);
    EXPECT_TRUE(s.mode == schedule_mode_t::unroll_switch);
    EXPECT_EQ(factors_of(s), (std::vector<int> {5}));

    const auto unit = parallel(runtime(), 1);
    EXPECT_TRUE(unit.mode == schedule_mode_t::unroll);
    EXPECT_EQ(factors_of(unit), (std::vector<int> {1}));
}

/// A coarser granularity ends in a one-granule stage instead of a switch
TEST(schedule_t, unbounded_data_parallel_coarse_granularity_is_pow2) {
    const auto s = parallel(runtime(), 4, 16);
    EXPECT_TRUE(s.mode == schedule_mode_t::unroll);
    EXPECT_EQ(factors_of(s), (std::vector<int> {4, 2, 1}));
    EXPECT_EQ(s.grain, 16);
}

/// A bound within the unroll factor needs only the one dispatch: a switch
TEST(schedule_t, bound_within_max_ur_is_a_switch) {
    const auto s = parallel(less_eq(3), 8);
    EXPECT_TRUE(s.mode == schedule_mode_t::switch_);
    EXPECT_EQ(factors_of(s), (std::vector<int> {3}));
    EXPECT_EQ(s.bound, 3);

    EXPECT_TRUE(parallel(less_eq(8), 8).mode == schedule_mode_t::switch_);
    EXPECT_TRUE(parallel(less_eq(9), 8).mode == schedule_mode_t::unroll_switch);
}

/// A bound under the unroll factor shrinks the stages to its own breakdown
TEST(schedule_t, bound_caps_the_stages) {
    {
        const std::vector<int> want = {5, 4, 2, 1};
        EXPECT_EQ(factors_of(reduce(less_eq(5), 8)), want);
    }
    {
        const std::vector<int> want = {8, 4, 2, 1};
        EXPECT_EQ(factors_of(reduce(less_eq(100), 8)), want);
    }
    {
        const std::vector<int> want = {3, 2, 1};
        EXPECT_EQ(factors_of(parallel(less_eq(6), 8, 2)), want);
    }
    {
        const std::vector<int> want = {4, 2, 1};
        EXPECT_EQ(factors_of(parallel(less_eq(20), 4, 2)), want);
    }
    EXPECT_EQ(factors_of(parallel(less_eq(7), 4, 16)), (std::vector<int> {1}));
}

/// A reduce bound is compared in granules, like the data-parallel one
TEST(schedule_t, reduce_bound_counts_granules) {
    EXPECT_EQ(factors_of(reduce(less_eq(8), 8, 4)), (std::vector<int> {2, 1}));
}

/// A literal that fits the unroll factor is a single literal dispatch
TEST(schedule_t, literal_within_max_ur_is_literal) {
    const auto s = reduce(size_parameter_t::literal(3), 8);
    EXPECT_TRUE(s.mode == schedule_mode_t::literal);
    EXPECT_EQ(factors_of(s), (std::vector<int> {3}));
    EXPECT_EQ(s.bound, 3);

    // The unroll factor itself is still one dispatch, reduce or parallel
    for (const auto &r : {reduce(size_parameter_t::literal(8), 8),
                 parallel(size_parameter_t::literal(8), 8)}) {
        EXPECT_TRUE(r.mode == schedule_mode_t::literal);
        EXPECT_EQ(factors_of(r), (std::vector<int> {8}));
    }
    const auto p = parallel(size_parameter_t::literal(3), 8);
    EXPECT_TRUE(p.mode == schedule_mode_t::literal);
    EXPECT_EQ(factors_of(p), (std::vector<int> {3}));
}

/// A literal the granularity divides is counted in granules
TEST(schedule_t, aligned_literal_counts_granules) {
    const auto s = reduce(size_parameter_t::literal(20), 4, 2);
    EXPECT_EQ(factors_of(s), (std::vector<int> {4, 2}));
    EXPECT_EQ(s.grain, 2);
}

/// A literal under two main stages is straight-line: main stage, then tail
TEST(schedule_t, literal_under_two_main_stages_is_straight_line) {
    const auto s = reduce(size_parameter_t::literal(11), 8);
    EXPECT_TRUE(s.mode == schedule_mode_t::literal);
    EXPECT_EQ(factors_of(s), (std::vector<int> {8, 3}));

    // 15 == 2 * 8 - 1 is the widest straight-line size
    {
        const auto mode = reduce(size_parameter_t::literal(15), 8).mode;
        EXPECT_TRUE(mode == schedule_mode_t::literal);
    }
    {
        const auto mode = reduce(size_parameter_t::literal(16), 8).mode;
        EXPECT_TRUE(mode == schedule_mode_t::unroll_literal);
    }
    {
        const std::vector<int> want = {8, 3};
        EXPECT_EQ(factors_of(reduce(size_parameter_t::literal(19), 8)), want);
    }
}

/// A literal the granularity does not divide keeps a runtime tail
TEST(schedule_t, unaligned_literal_keeps_a_runtime_tail) {
    const auto s = parallel(size_parameter_t::literal(19), 4, 2);
    EXPECT_TRUE(s.mode == schedule_mode_t::unroll);
    EXPECT_EQ(factors_of(s), (std::vector<int> {4, 2, 1}));
    EXPECT_EQ(s.bound, 19);

    const auto r = reduce(size_parameter_t::literal(19), 4, 2);
    EXPECT_EQ(factors_of(r), (std::vector<int> {4, 1}));
}

/// A unit stage dispatches once per granule, a wider one while a whole fits
TEST(schedule_t, stage_trips_count_dispatches) {
    using loop::schedule_t;
    EXPECT_EQ(schedule_t::stage_trips(1, 1, 5), 5);
    EXPECT_EQ(schedule_t::stage_trips(1, 4, 5), 2);
    EXPECT_EQ(schedule_t::stage_trips(1, 4, 4), 1);
    EXPECT_EQ(schedule_t::stage_trips(4, 1, 7), 1);
    EXPECT_EQ(schedule_t::stage_trips(4, 1, 8), 2);
    EXPECT_EQ(schedule_t::stage_trips(4, 2, 7), 0);
}

/// A stage leaves less than its own width, none for a unit stage
TEST(schedule_t, stage_exit_bounds_the_remainder) {
    using loop::schedule_t;
    EXPECT_EQ(schedule_t::stage_exit(4, 1, 100), 3);
    EXPECT_EQ(schedule_t::stage_exit(4, 2, 100), 7);
    EXPECT_EQ(schedule_t::stage_exit(4, 1, 2), 2);
    EXPECT_EQ(schedule_t::stage_exit(1, 8, 100), 0);
}

} // namespace rvjit
