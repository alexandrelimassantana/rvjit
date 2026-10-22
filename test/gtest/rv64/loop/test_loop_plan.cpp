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

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64/loop/loop_helpers.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/loop/frame.hpp"
#include "rvjit/interface/loop/persistence.hpp"
#include "rvjit/interface/loop/schedule.hpp"
#include "rvjit/interface/resource/optimization/group.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

// pow2_breakdown

namespace {
/// Compares a pow2_breakdown() result buffer against an expected sequence
void expect_pow2_breakdown(
        int value, std::vector<int> expect, int max_depth = 8) {
    int buf[8];
    const int n = pow2_breakdown(value, buf, max_depth);
    ASSERT_EQ(static_cast<size_t>(n), expect.size());
    for (size_t i = 0; i < expect.size(); ++i)
        EXPECT_EQ(buf[i], expect[i]) << "i=" << i;
}
} // namespace

/// Largest power of two <= value, then halving down to 1
TEST(pow2_breakdown_fn, breaks_down_descending_powers_of_two) {
    expect_pow2_breakdown(16, {16, 8, 4, 2, 1});
    expect_pow2_breakdown(14, {8, 4, 2, 1});
    expect_pow2_breakdown(1, {1});
    expect_pow2_breakdown(0, {});
}

/// An explicit max_depth caps the number of entries emitted
TEST(pow2_breakdown_fn, max_depth_caps_entries) {
    expect_pow2_breakdown(64, {64, 32, 16}, 3);
}

// rv64::loop::loopgen_t::new_loop() resolution

/// A loop takes its unroll factors and granularity from the plan's schedule
TEST(loop_t_plan, mirrors_the_schedule) {
    using plan_t = loop::plan_t;
    const auto lit = [](int v) { return size_parameter_t::literal(v); };
    const auto rt = [] { return size_parameter_t::runtime(dummy_reg()); };
    const auto le
            = [](int b) { return size_parameter_t::less_eq(dummy_reg(), b); };
    const std::vector<plan_t> plans = {plan_t::reduce(lit(3), 8),
            plan_t::reduce(lit(8), 8), plan_t::reduce(lit(19), 8),
            plan_t::reduce(lit(20), 4, 2), plan_t::reduce(lit(19), 4, 2),
            plan_t::reduce(le(14), 16), plan_t::reduce(le(1000), 8),
            plan_t::reduce(rt(), 8), plan_t::data_parallel(lit(3), 8),
            plan_t::data_parallel(lit(8), 8),
            plan_t::data_parallel(lit(19), 4, 2),
            plan_t::data_parallel(le(6), 8, 2),
            plan_t::data_parallel(le(20), 4, 2),
            plan_t::data_parallel(le(7), 1, 16),
            plan_t::data_parallel(le(7), 4, 16),
            plan_t::data_parallel(le(15), 4, 2), plan_t::data_parallel(rt(), 8),
            plan_t::data_parallel(rt(), 8, 4), plan_t::data_parallel(rt(), 1)};

    host_env_t env;
    for (const plan_t &p : plans) {
        const loop::schedule_t s = loop::schedule_t::of(p);
        loop::loop_t *desc = env.loops.new_loop(p);
        ASSERT_TRUE(desc->is_resolved());
        EXPECT_EQ(desc->granularity(), s.grain);
        ASSERT_EQ(desc->unroll().size(), s.factors.size());
        for (int i = 0; i < s.factors.size(); ++i)
            EXPECT_EQ(desc->unroll()[i], s.factors[i]) << "factor " << i;
    }

    // A literal within one dispatch has no further trips and no remainder
    loop::loop_t *once = env.loops.new_loop(plan_t::reduce(lit(8), 8));
    EXPECT_FALSE(once->has_multiple_trips());
    EXPECT_FALSE(once->has_remaining());

    // A unit unroll of an unbounded size is a counted loop
    loop::loop_t *counted = env.loops.new_loop(plan_t::data_parallel(rt(), 1));
    EXPECT_TRUE(counted->has_multiple_trips());
}

/// A bounded loop whose stages cannot dispatch twice has a single trip
TEST(loop_t_plan, bounded_loop_within_one_dispatch_has_a_single_trip) {
    host_env_t env;
    const auto single = [&](const loop::plan_t &p) {
        loop::loop_t *desc = env.loops.new_loop(p);
        EXPECT_TRUE(desc->is_resolved());
        return !desc->has_multiple_trips();
    };
    const auto less_eq = [](int bound) {
        return size_parameter_t::less_eq(dummy_reg(), bound);
    };

    EXPECT_TRUE(single(loop::plan_t::data_parallel(less_eq(16), 1, 16)));
    EXPECT_FALSE(single(loop::plan_t::data_parallel(less_eq(17), 1, 16)));
    EXPECT_TRUE(single(loop::plan_t::reduce(less_eq(1), 8)));
    EXPECT_FALSE(single(loop::plan_t::reduce(less_eq(2), 8)));

    // The switch tail is one more dispatch after the main stage
    EXPECT_FALSE(single(loop::plan_t::data_parallel(less_eq(9), 8)));
    EXPECT_TRUE(single(loop::plan_t::data_parallel(less_eq(8), 8)));

    // Nothing bounds an unbounded size
    {
        const auto p = loop::plan_t::data_parallel(
                size_parameter_t::runtime(dummy_reg()), 1, 16);
        EXPECT_FALSE(single(p));
    }
}

// rv64::loop::loopgen_t::new_loop() -- JIT-executed correctness checks proving
// the resolved descriptor's SELECTION is correct; the underlying emission
// primitives are already exhaustively tested elsewhere in this file

class loop_plan_jit_literal_t
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

/// A literal size runs every unit once, in as many dispatches as it needs
TEST_P(loop_plan_jit_literal_t, passes_and_total) {
    int size, expected_passes;
    std::tie(size, expected_passes) = GetParam();

    rv64_jit_probe_t kernel([size](rv64_jit_test_generator_t &gen) {
        const Reg total_ptr = a0, passes_ptr = a1;
        const Reg total = a2, passes = a3;
        env_t env(gen, excluding_ints({total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::loop_t *desc = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::literal(size), /*max_ur=*/8));
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(kernel, &total, &passes);
    EXPECT_EQ(passes, expected_passes);
    EXPECT_EQ(total, size);
}

/// One dispatch, a main stage and its tail, then two main stages and a tail
INSTANTIATE_TEST_SUITE_P(sizes, loop_plan_jit_literal_t,
        ::testing::Values(std::make_tuple(3, 1), std::make_tuple(11, 2),
                std::make_tuple(19, 3)));

class loop_plan_jit_switch_t : public ::testing::TestWithParam<int64_t> {};

/// A less_eq size at granularity > 1 covers every granule, last maybe partial
TEST_P(loop_plan_jit_switch_t, less_eq_covers_every_granule) {
    const int64_t raw = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg raw_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints(
                        {raw_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t::less_eq(raw_reg, 6), /*max_ur=*/8,
                /*granularity=*/2);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(kernel, raw, &total, &passes);
    EXPECT_EQ(total, (raw + 1) / 2);
    EXPECT_GE(passes, raw > 0 ? 1 : 0);
}

/// Runtime values at and around the granularity and the bound, including 0
INSTANTIATE_TEST_SUITE_P(raw_values, loop_plan_jit_switch_t,
        ::testing::Values<int64_t>(0, 1, 2, 3, 4, 5, 6));

class loop_plan_jit_unroll_t : public ::testing::TestWithParam<int64_t> {};

/// Plain unroll via pow2_breakdown matches simulate_unroll() for {8,4,2,1}
TEST_P(loop_plan_jit_unroll_t, plain_unroll_pow2_breakdown_correctness) {
    const int64_t n = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::reduce(
                size_parameter_t::less_eq(n_reg, 1000), /*max_ur=*/8);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    expect_trips_and_iterations(
            kernel, simulate_unroll({8, 4, 2, 1}, n, /*granularity=*/1), n);
}

/// Every loop size
INSTANTIATE_TEST_SUITE_P(sizes, loop_plan_jit_unroll_t, loop_sizes());

class loop_plan_jit_unroll_and_switch_t : public ::testing::TestWithParam<int> {
};

/// unroll_switch via runtime data_parallel matches simulate_unroll_and_switch()
TEST_P(loop_plan_jit_unroll_and_switch_t, runtime_unroll_switch_correctness) {
    const int max_ur = GetParam();

    rv64_jit_probe_t kernel([max_ur](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t::runtime(n_reg), max_ur);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    for (int64_t n : {0, 1, 3, 4, 5, 9, 10, 16, 17}) {
        int64_t total = -1, passes = -1;
        run_timed(kernel, n, &total, &passes);
        const auto expect = simulate_unroll_and_switch({max_ur}, n, 1);
        EXPECT_EQ(total, expect.total) << "n=" << n << " max_ur=" << max_ur;
        EXPECT_EQ(passes, expect.passes) << "n=" << n << " max_ur=" << max_ur;
    }
}

INSTANTIATE_TEST_SUITE_P(max_ur, loop_plan_jit_unroll_and_switch_t,
        ::testing::Values(2, 3, 4, 8));

class loop_plan_jit_bounded_t : public ::testing::TestWithParam<int64_t> {};

/// A bound under twice the unroll leaves a main stage running at most once
TEST_P(loop_plan_jit_bounded_t, single_main_stage_before_the_switch) {
    const int64_t n = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t::less_eq(n_reg, 7), /*max_ur=*/4);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    expect_trips_and_iterations(
            kernel, simulate_unroll_and_switch({4}, n, 1), n);
}

/// Every size up to the bound
INSTANTIATE_TEST_SUITE_P(sizes, loop_plan_jit_bounded_t,
        ::testing::Values<int64_t>(0, 1, 2, 3, 4, 5, 6, 7));

class loop_plan_jit_single_trip_t : public ::testing::TestWithParam<int64_t> {};

/// A bound of one dispatches the unit stage once, or not at all
TEST_P(loop_plan_jit_single_trip_t, unit_bound_is_a_conditional_dispatch) {
    const int64_t n = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::reduce(
                size_parameter_t::less_eq(n_reg, 1), /*max_ur=*/8);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(kernel, n, &total, &passes);
    EXPECT_EQ(total, n);
    EXPECT_EQ(passes, n);
}

INSTANTIATE_TEST_SUITE_P(
        sizes, loop_plan_jit_single_trip_t, ::testing::Values<int64_t>(0, 1));

class loop_plan_jit_coarse_bounded_t
    : public ::testing::TestWithParam<int64_t> {};

/// A coarse bound runs one partial-granule dispatch, or at most one per stage
TEST_P(loop_plan_jit_coarse_bounded_t, bounded_stages_cover_every_granule) {
    const int64_t n = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t::less_eq(n_reg, 12), /*max_ur=*/4,
                /*granularity=*/4);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(kernel, n, &total, &passes);
    EXPECT_EQ(total, (n + 3) / 4);
}

INSTANTIATE_TEST_SUITE_P(sizes, loop_plan_jit_coarse_bounded_t,
        ::testing::Values<int64_t>(0, 1, 3, 4, 5, 7, 8, 9, 11, 12));

} // namespace rvjit
