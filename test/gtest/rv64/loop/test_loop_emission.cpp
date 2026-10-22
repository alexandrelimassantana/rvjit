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
#include "rvjit/interface/resource/optimization/group.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// A less_eq size off the granularity is drained by a one-granule stage
TEST(control_flow_unroll_and_switch,
        misaligned_n_is_drained_by_the_tail_stage) {
    const int granularity = 4;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::loop_t *desc = env.loops.new_loop(loop::plan_t::data_parallel(
                size_parameter_t::less_eq(n_reg, 16),
                /*max_ur=*/4, /*granularity=*/4));
        desc->allocate();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb * 4);
        });
        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    for (int64_t n : {1, 3, 4, 5, 9, 10, 11, 13, 14, 15, 16}) {
        int64_t total = -1, passes = -1;
        run_timed(kernel, n, &total, &passes);
        {
            const auto want = (n + granularity - 1) / granularity * granularity;
            EXPECT_EQ(total, want) << "n=" << n;
        }
    }
}

// loop::loop_t::generate(): the frame-driving dispatcher

// A caller binds variables to a loop via loop_t::bind() after new_loop()
// resolves it; loop() then drives them once per `body` call. The dispatcher assumes
// exactly one work unit per call, so every suite below fixes unroll
// factors to 1 -- literal via a single-element literal, switch_ trivially
// (it always dispatches exactly once), and unroll via a `runtime`-kind
// parameter (always resolves to a single {1} factor)

/// A step set through set_step() after binding is the one the frame emits
TEST(unrolled_loop_frame, step_set_after_bind_is_seen) {
    static const int64_t start = 10;
    static const int delta = 7;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        const auto entry = env.pool.new_variable(source_t::adopt(r));
        loop::plan_t plan = loop::plan_t::reduce(
                size_parameter_t::literal(1), /*max_ur=*/8);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->frame().bind(entry, loop::bind_t::reloaded());
        desc->frame().set_step(entry, step_t::plain(const_t(delta)));
        desc->allocate();
        desc->generate([&](int) {});

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, &out);
    EXPECT_EQ(out, start + delta);
}

class unrolled_loop_frame_switch_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

/// switch_ mode dispatches once, so the frame steps once for any id
TEST_P(unrolled_loop_frame_switch_t, steps_exactly_once) {
    int max_n;
    int64_t id;
    std::tie(max_n, id) = GetParam();
    static const int64_t start = 10;
    static const int delta = 3;

    rv64_jit_probe_t kernel([max_n](rv64_jit_test_generator_t &gen) {
        const Reg id_reg = a0, out_ptr = a1;
        env_t env(gen, excluding_ints({id_reg, out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        const auto entry = env.pool.new_variable(source_t::adopt(r));
        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t::less_eq(id_reg, max_n), max_n);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->frame().bind(entry,
                loop::bind_t::reloaded().stepped(
                        step_t::plain(const_t(delta))));
        desc->allocate();
        desc->generate([&](int) {});

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, id, &out);
    EXPECT_EQ(out, start + delta);
}

/// Direct, if/else, and jump table dispatch paths
INSTANTIATE_TEST_SUITE_P(n_and_id, unrolled_loop_frame_switch_t,
        ::testing::Values(std::make_tuple(1, 1), std::make_tuple(3, 1),
                std::make_tuple(3, 2), std::make_tuple(3, 3)));

class unrolled_loop_frame_unroll_t : public ::testing::TestWithParam<int64_t> {
};

/// unroll mode with a `runtime` size runs one frame pass per element
TEST_P(unrolled_loop_frame_unroll_t, steps_once_per_element) {
    const int64_t n = GetParam();
    static const int64_t start = 10;
    static const int delta = 3;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, out_ptr = a1;
        env_t env(gen, excluding_ints({n_reg, out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        const auto entry = env.pool.new_variable(source_t::adopt(r));
        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t(n_reg), /*max_ur=*/1);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->frame().bind(entry,
                loop::bind_t::reloaded().stepped(
                        step_t::plain(const_t(delta))));
        desc->allocate();
        desc->generate([&](int) {});

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, n, &out);
    EXPECT_EQ(out, start + delta * n);
}

/// Every loop size
INSTANTIATE_TEST_SUITE_P(sizes, unrolled_loop_frame_unroll_t, loop_sizes());

class unrolled_loop_frame_clobber_t : public ::testing::TestWithParam<int64_t> {
};

/// A slot-backed binding is recovered on every dispatch despite the clobber
TEST_P(unrolled_loop_frame_clobber_t, recovers_across_passes_despite_clobber) {
    const int64_t n = GetParam();

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, out_ptr = a1;
        env_t env(gen, excluding_ints({n_reg, out_ptr}));

        const Reg a = Reg(env.pool.integers().pop());
        const Reg b = Reg(env.pool.integers().pop());
        gen.li(a, 0);

        const auto entry = env.pool.new_variable(source_t::adopt(a));
        loop::plan_t plan = loop::plan_t::data_parallel(
                size_parameter_t(n_reg), /*max_ur=*/1);
        loop::loop_t *desc = env.loops.new_loop(plan);
        desc->frame().bind(entry,
                loop::bind_t::given(slot_t(b)).stepped(
                        step_t::plain(const_t(1))));
        desc->allocate();
        desc->generate([&](int) {
            gen.li(a, -1); // clobber -- the frame's restore() must recover it
        });

        gen.sd(a, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, n, &out);
    EXPECT_EQ(out, n);
}

/// Every loop size
INSTANTIATE_TEST_SUITE_P(sizes, unrolled_loop_frame_clobber_t, loop_sizes());

// rv64::loop::loop_t / loop::unroll_t state (non-codegen)

/// is_valid() accepts a non-empty, strictly descending list of factors >= 1
TEST(unroll_t_state, is_valid_checks_bounds_and_descending_order) {
    EXPECT_FALSE(loop::unroll_t().is_valid()); // empty
    EXPECT_TRUE(loop::unroll_t(1).is_valid());
    EXPECT_TRUE((loop::unroll_t {4, 2, 1}).is_valid());
    {
        // Ascending, not descending
        const loop::unroll_t ascending {1, 2};
        EXPECT_FALSE(ascending.is_valid());
    }
    EXPECT_FALSE((loop::unroll_t {0}).is_valid()); // factor < 1
}

/// need_unrolling() reflects whether the descriptor's own loop::unroll_t is valid
TEST(loop_t_state, need_unrolling_reflects_unroll_t_validity) {
    rv64::loop::loop_t desc = make_desc(rv64::loop::loop_t::mode_t::unroll);
    EXPECT_TRUE(desc.need_unrolling()); // no factors set
    desc.factors = loop::unroll_t(1);
    EXPECT_FALSE(desc.need_unrolling());
}

/// literal mode never needs a `counter` or `tmp`
TEST(loop_t_state, literal_mode_never_needs_buffers) {
    rv64::loop::loop_t desc = literal_desc();
    EXPECT_TRUE(desc.demand().is_none());
}

/// is_valid() rejects an `iter`/`tmp` alias, regardless of mode
TEST(loop_t_state, is_valid_rejects_aliased_iter_and_tmp) {
    rv64::loop::loop_t desc = unrolled_desc(1);
    desc.counter = a0;
    desc.tmp = a0; // aliased -- invalid regardless of mode
    EXPECT_FALSE(desc.is_valid());
    desc.tmp = a1;
    EXPECT_TRUE(desc.is_valid());
}

/// is_valid() rejects a granularity below 1
TEST(loop_t_state, is_valid_rejects_nonpositive_granularity) {
    rv64::loop::loop_t desc = unrolled_desc(1);
    desc.counter = a0;
    desc.tmp = a1;
    desc.grain = 0;
    EXPECT_FALSE(desc.is_valid());
    desc.grain = 1;
    EXPECT_TRUE(desc.is_valid());
}

/// is_valid() enforces switch_'s own NCASES bound on the unrolling
TEST(loop_t_state, is_valid_enforces_switch_ncases_bound) {
    rv64::loop::loop_t desc
            = switch_desc(operation::control_flow_t::NCASES + 1);
    desc.counter = a0;
    desc.tmp = a1;
    EXPECT_FALSE(desc.is_valid());

    desc.factors = loop::unroll_t(operation::control_flow_t::NCASES);
    EXPECT_TRUE(desc.is_valid());
}

/// is_valid() enforces unroll_switch's NCASES bound on the unrolling
TEST(loop_t_state, is_valid_enforces_unroll_and_switch_ncases_bound) {
    rv64::loop::loop_t desc
            = unroll_switch_desc(operation::control_flow_t::NCASES + 2);
    desc.counter = a0;
    desc.tmp = a1;
    EXPECT_FALSE(desc.is_valid());

    desc.factors = loop::unroll_t(operation::control_flow_t::NCASES + 1);
    EXPECT_TRUE(desc.is_valid());
}

} // namespace rvjit
