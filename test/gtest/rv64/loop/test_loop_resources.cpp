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
#include "rvjit/interface/resource/optimization/group.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

// rv64::loop::loop_t resources (materializable_t)

namespace {

/// Integer registers `d` draws on its next allocation
int ints_of(const materializable_t &d) {
    return d.demand()[resource_t::integers];
}

} // namespace

struct demand_row_t {
    rv64::loop::loop_t::mode_t mode;
    std::vector<int> factors;
    int grain;
    bool counter; // supplied by the caller
    bool tmp; // supplied by the caller
    int ints; // what the loop draws
};

class loop_t_demand : public ::testing::TestWithParam<demand_row_t> {};

/// demand() counts the integers each mode draws, and allocate() draws them
TEST_P(loop_t_demand, matches_what_allocate_draws) {
    const demand_row_t row = GetParam();
    host_env_t env;
    const Reg counter = row.counter ? Reg(env.pool.integers().pop()) : Reg();
    const Reg tmp = row.tmp ? Reg(env.pool.integers().pop()) : Reg();
    const loop::unroll_t factors(row.factors.begin(), row.factors.end());
    rv64::loop::loop_t d = make_desc(row.mode, factors, &env);
    d.grain = row.grain;
    d.counter = counter;
    d.tmp = tmp;
    EXPECT_EQ(ints_of(d), row.ints);

    const int before = env.pool.integers().available();
    EXPECT_TRUE(d.allocate().ok());
    EXPECT_EQ(before - env.pool.integers().available(), row.ints);
    EXPECT_TRUE(d.is_valid());
}

using loop_mode_t = rv64::loop::loop_t::mode_t;

/// Literal, switch_, unroll (folding or not), unroll_switch, unroll_literal
INSTANTIATE_TEST_SUITE_P(modes, loop_t_demand,
        ::testing::Values(demand_row_t {loop_mode_t::literal, {3}, 1, 0, 0, 0},
                demand_row_t {loop_mode_t::switch_, {4}, 1, 0, 0, 2},
                demand_row_t {loop_mode_t::switch_, {4}, 1, 1, 0, 1},
                demand_row_t {loop_mode_t::switch_, {4}, 1, 0, 1, 1},
                demand_row_t {loop_mode_t::switch_, {4}, 1, 1, 1, 0},
                demand_row_t {loop_mode_t::unroll, {1}, 4, 0, 0, 1},
                demand_row_t {loop_mode_t::unroll, {1}, 4096, 0, 0, 2},
                demand_row_t {loop_mode_t::unroll, {1}, 4, 1, 0, 0},
                demand_row_t {loop_mode_t::unroll, {2, 1}, 4, 0, 0, 2},
                demand_row_t {loop_mode_t::unroll, {4, 1}, 1, 0, 0, 2},
                demand_row_t {loop_mode_t::unroll, {4, 1}, 1, 1, 0, 1},
                demand_row_t {loop_mode_t::unroll, {4, 1}, 1, 1, 1, 0},
                demand_row_t {loop_mode_t::unroll_switch, {1}, 1, 0, 0, 2},
                demand_row_t {loop_mode_t::unroll_switch, {4}, 1, 0, 0, 2},
                demand_row_t {
                        loop_mode_t::unroll_literal, {8, 3}, 1, 0, 0, 2}));

/// A query through the base type sees direct edits of the descriptor
TEST(loop_t_state, base_reference_sees_register_assignments) {
    rv64::loop::loop_t d = unrolled_desc(loop::unroll_t {4, 1});
    const materializable_t &base = d;
    EXPECT_EQ(ints_of(base), 2);
    EXPECT_FALSE(base.is_valid());

    d.counter = a0;
    d.tmp = a1;
    EXPECT_EQ(ints_of(base), 0);
    EXPECT_TRUE(base.is_valid());
}

/// allocate() draws counter then tmp, keeps assigned ones, redraws each call
TEST(loop_t_state, allocate_draws_counter_then_tmp) {
    host_env_t env;
    const auto make = [&] {
        rv64::loop::loop_t d(
                env.e, env.arith, env.cf, env.pool, env.stack, env.xfer);
        d.factors = loop::unroll_t {4, 1};
        return d;
    };
    env.pool.scope([&] {
        rv64::loop::loop_t d = make();
        EXPECT_TRUE(d.allocate().ok());
        EXPECT_TRUE(d.counter == Reg(env.pool.integers().id(0)));
        EXPECT_TRUE(d.tmp == Reg(env.pool.integers().id(1)));
        EXPECT_TRUE(d.is_valid());
        EXPECT_EQ(ints_of(d), 2); // the next call draws them again
        return shortfall_t {};
    });

    env.pool.scope([&] {
        rv64::loop::loop_t d = make();
        d.counter = a0;
        EXPECT_EQ(ints_of(d), 1);
        EXPECT_TRUE(d.allocate().ok());
        EXPECT_TRUE(d.counter == a0);
        EXPECT_TRUE(is_not_zero(d.tmp));
        EXPECT_TRUE(d.supplied().has(reg_name(a0)));
        EXPECT_EQ(ints_of(d), 1);
        return shortfall_t {};
    });
}

/// remaining() reports the counter register
TEST(loop_t_state, remaining_is_the_counter) {
    rv64::loop::loop_t d = unrolled_desc(1);
    EXPECT_FALSE(is_not_zero(d.remaining()));
    d.counter = a3;
    EXPECT_TRUE(d.remaining() == a3);
}

/// Each mode lends `tmp` first, then the spilled counter, one integer each
TEST(loop_t_state, lending_levels_per_mode) {
    EXPECT_EQ(literal_desc().levels(), 1);
    EXPECT_EQ(switch_desc(4).levels(), 1);

    rv64::loop::loop_t u = unrolled_desc(loop::unroll_t {4, 1});
    EXPECT_EQ(u.levels(), 3);
    EXPECT_EQ(u.relief(0, 1)[resource_t::integers], 1);
    EXPECT_EQ(u.relief(0, 2)[resource_t::integers], 2);
    EXPECT_EQ(u.relief(0, 2)[resource_t::stack], -1); // the spill slot
    EXPECT_EQ(u.relief(2, 0)[resource_t::integers], -2);

    u.shared = true; // a shared counter cannot be spilled
    EXPECT_EQ(u.levels(), 2);
    EXPECT_EQ(u.relief(0, 1)[resource_t::stack], 0);
}

/// A dropped transaction returns the loop's registers to the pool
TEST(control_flow_jit, dropped_transaction_returns_loop_registers) {
    host_env_t env;
    rv64::loop::loop_t desc(
            env.e, env.arith, env.cf, env.pool, env.stack, env.xfer);
    desc.factors = loop::unroll_t {4, 1};
    const int before = env.pool.integers().available();
    env.pool.begin();
    EXPECT_TRUE(desc.allocate().ok());
    const int during = env.pool.integers().available();
    env.pool.drop();
    EXPECT_EQ(before - during, 2);
    EXPECT_EQ(env.pool.integers().available(), before);
}

// loop::loopgen_t buffer reuse and aliasing

/// new_loop() resolves without allocating; allocate() draws in the scope
TEST(control_flow_jit, new_loop_allocates_nothing_until_allocate) {
    host_env_t env;
    const int before = env.pool.integers().available();
    loop::loop_t *desc = env.loops.new_loop(loop::plan_t::reduce(
            size_parameter_t::runtime(dummy_reg()), /*max_ur=*/8));
    ASSERT_TRUE(desc->is_resolved());
    EXPECT_FALSE(desc->is_valid());
    EXPECT_EQ(env.pool.integers().available(), before);

    env.pool.scope([&] {
        EXPECT_TRUE(desc->allocate().ok());
        EXPECT_TRUE(desc->is_valid());
        EXPECT_EQ(env.pool.integers().available(), before - 2);
        return shortfall_t {};
    });
}

/// set_level() lends tmp, then the spilled counter, absolutely; literal none
TEST(control_flow_jit, lend_levels_counts_and_idempotence) {
    int levels = -1, n_both = -1, n_one = -1, n_none = -1, n_literal = -1;
    int offset_levels = -1, offset_allocated = -1, level_after = -1;
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        loop::loop_t *desc = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(dummy_reg()), /*max_ur=*/8));
        levels = desc->levels();

        desc->set_level(2);
        desc->set_level(2); // idempotent, and draws no slot
        offset_levels = env.stack.new_stack_slot().offset;
        desc->allocate();
        n_both = desc->released().size();
        offset_allocated = env.stack.new_stack_slot().offset;

        desc->set_level(1);
        n_one = desc->released().size();
        level_after = desc->level();
        desc->set_level(0);
        n_none = desc->released().size();

        loop::loop_t *lit = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::literal(1), /*max_ur=*/8));
        n_literal = lit->levels();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_EQ(levels, 3);
    EXPECT_EQ(offset_levels, 0);
    EXPECT_EQ(n_both, 2);
    EXPECT_EQ(offset_allocated, 16); // the spill slot sits in between
    EXPECT_EQ(n_one, 1);
    EXPECT_EQ(level_after, 1);
    EXPECT_EQ(n_none, 0);
    EXPECT_EQ(n_literal, 1);
}

/// alias() adopts the source loop's registers, and both loops dispatch right
TEST(control_flow_jit, alias_shares_registers_and_dispatches) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg n_reg = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints({n_reg, total_ptr, passes_ptr, total, passes}));
        gen.mv(total, x0);
        gen.mv(passes, x0);

        // A `runtime` reduce-kind loop resolves to `unroll{4,2,1}` (needs
        // both counter and tmp, since the ur=4 stage requires a threshold
        // register) -- unlike a `literal` loop, which needs neither
        loop::loop_t *tail = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_reg), /*max_ur=*/4));

        // A `runtime` data_parallel-kind loop, granularity 1, resolves to
        // `unroll_switch{4}` instead -- a differently-shaped loop from
        // `tail`, still sharing the same counter/tmp registers
        loop::loop_t *main = env.loops.alias(
                loop::plan_t::data_parallel(
                        size_parameter_t::runtime(n_reg), /*max_ur=*/4),
                *tail);

        const int before = env.pool.integers().available();
        ASSERT_TRUE(tail->allocate().ok());
        const int after_tail = env.pool.integers().available();
        ASSERT_TRUE(main->allocate().ok());
        const int after_main = env.pool.integers().available();

        // main adopts tail's own registers instead of allocating its own
        EXPECT_EQ(after_main, after_tail);
        EXPECT_LT(after_tail, before);

        ASSERT_TRUE(tail->is_valid());
        ASSERT_TRUE(main->is_valid());

        main->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
        });

        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(kernel, static_cast<int64_t>(4), &total, &passes);
    // main resolves to unroll_switch{4}; n=4 drains via the main stage in
    // a single pass, leaving nothing for the switch_case tail
    EXPECT_EQ(passes, 1);
    EXPECT_EQ(total, 4);
}

class control_flow_reuse_buffer_t : public ::testing::TestWithParam<int64_t> {};

/// Lent registers (tmp, spilled counter) may be clobbered by the body safely
TEST_P(control_flow_reuse_buffer_t, lent_buffers_survive_a_clobbering_body) {
    const int64_t n = GetParam();
    const int max_ur = 4; // pow2_breakdown(4) == {4, 2, 1}

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg remaining = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints(
                        {remaining, total_ptr, passes_ptr, total, passes}));

        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::loop_t *desc = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(remaining), /*max_ur=*/4));
        desc->set_level(2);
        desc->allocate();
        const buffer::buffer_t lent = desc->released();

        env.stack.reserve_stack();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
            for (int i = 0; i < lent.size(); ++i)
                gen.li(Reg(lent.id(i)), -1); // clobber every lent register
        });
        env.stack.release_stack();

        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    expect_trips_and_iterations(
            kernel, simulate_unroll(pow2_factors(max_ur), n, 1), n);
}

INSTANTIATE_TEST_SUITE_P(sizes, control_flow_reuse_buffer_t, loop_sizes());

class loop_stage_width_t
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

/// Stage widths in and past simm12 step correctly, lent `tmp` clobbered or not
TEST_P(loop_stage_width_t, steps_every_granule) {
    int grain, level;
    std::tie(grain, level) = GetParam();
    const int max_ur = 4;

    rv64_jit_probe_t kernel([grain, level](rv64_jit_test_generator_t &gen) {
        const Reg remaining = a0, total_ptr = a1, passes_ptr = a2;
        const Reg total = a3, passes = a4;
        env_t env(gen,
                excluding_ints(
                        {remaining, total_ptr, passes_ptr, total, passes}));

        gen.mv(total, x0);
        gen.mv(passes, x0);

        loop::loop_t *desc = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(remaining), max_ur, grain));
        desc->set_level(level);
        desc->allocate();
        const buffer::buffer_t lent = desc->released();

        env.stack.reserve_stack();
        desc->generate([&](int nb) {
            gen.addi(passes, passes, 1);
            gen.addi(total, total, nb);
            for (int i = 0; i < lent.size(); ++i)
                gen.li(Reg(lent.id(i)), -1);
        });
        env.stack.release_stack();

        gen.sd(total, total_ptr, 0);
        gen.sd(passes, passes_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    for (int64_t k : {0, 1, 2, 3, 4, 7, 8, 9}) {
        for (int64_t r : {0, grain / 2}) {
            const int64_t n = k * grain + r;
            if (n == 0 && r != 0) continue;
            const auto expect = simulate_unroll(pow2_factors(max_ur), n, grain);
            const int64_t granules = (n + grain - 1) / grain;
            int64_t total = -1, passes = -1;
            run_timed(kernel, n, &total, &passes);
            EXPECT_EQ(passes, expect.passes) << "n=" << n;
            EXPECT_EQ(total, granules) << "n=" << n;
        }
    }
}

/// Widths 4/2 * 1024 overflow simm12, 4 * 600 only the main stage's
INSTANTIATE_TEST_SUITE_P(grains_and_levels, loop_stage_width_t,
        ::testing::Combine(
                ::testing::Values(1, 600, 1024), ::testing::Values(0, 2)));

// loop lending

/// A dropped loop transaction rewinds the lending level and the spill slot
TEST(loop_lending, drop_rewinds_the_level) {
    host_env_t env;
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::runtime(a0), /*max_ur=*/8));
    ASSERT_TRUE(h->set_level(1));
    ASSERT_TRUE(h->allocate().ok());

    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(h->set_level(2));
    EXPECT_EQ(h->level(), 2);
    ASSERT_TRUE(env.loops.drop());
    EXPECT_EQ(h->level(), 1);
    EXPECT_EQ(h->released().size(), 1);

    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(h->set_level(2));
    ASSERT_TRUE(env.loops.commit());
    EXPECT_EQ(h->level(), 2);
}

/// A loop's lent registers are drawn first, and only, in the scope given them
TEST(loop_lending, released_registers_are_given_to_the_body_scope) {
    host_env_t env;
    loop::loop_t *outer = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::runtime(a0), /*max_ur=*/4));
    loop::loop_t *inner = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::runtime(a1), /*max_ur=*/4));

    optimization::group_t nest(optimization::order_t::level_first);
    nest.declare(*outer, *inner);
    ASSERT_EQ(nest.levels(), 5); // none, tmp x2, counter x2
    ASSERT_TRUE(nest.set_level(2)); // both tmps

    auto &ints = env.pool.integers();
    ASSERT_TRUE(outer->allocate().ok());
    const int base = ints.available();
    ASSERT_EQ(outer->released().size(), 1);

    env.pool.scope(outer->released(), [&] {
        EXPECT_EQ(ints.available(), base + 1);
        EXPECT_TRUE(inner->allocate().ok());
        EXPECT_FALSE((inner->supplied() & outer->released()).is_empty());
        env.pool.scope(inner->released(), [&] {
            EXPECT_EQ(Reg(ints.pop()), Reg(inner->released().id(0)));
            return shortfall_t {};
        });
        return shortfall_t {};
    });
    // What the body drew is free again once it closes
    EXPECT_EQ(ints.available(), base);
}

/// A transient after an inner loop reuses its registers; the loops count right
TEST(loop_lending, transient_after_an_inner_loop_reuses_its_registers) {
    int r_id = -1;
    buffer::buffer_t inner_regs;
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        const Reg n_outer = a0, n_inner = a1, out = a2;
        const Reg inner_total = a3, outer_total = a4;
        env_t env(gen,
                excluding_ints(
                        {n_outer, n_inner, out, inner_total, outer_total}));

        gen.mv(inner_total, x0);
        gen.mv(outer_total, x0);

        loop::loop_t *outer = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_outer), /*max_ur=*/4));
        loop::loop_t *inner = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_inner), /*max_ur=*/4));
        Reg r;
        outer->allocate();
        env.pool.scope([&] {
            env.pool.scope([&] { return inner->allocate(); });
            inner_regs = inner->supplied();
            env.pool.scope([&] {
                r = Reg(env.pool.integers().pop());
                return shortfall_t {};
            });
            return shortfall_t {};
        });
        r_id = reg_name(r);

        outer->generate([&](int ob) {
            inner->generate(
                    [&](int ib) { gen.addi(inner_total, inner_total, ib); });
            gen.li(r, ob); // after the inner loop's dispatch
            gen.add(outer_total, outer_total, r);
        });
        gen.sd(inner_total, out, 0);
        gen.sd(outer_total, out, 8);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_TRUE(inner_regs.has(r_id));

    const int64_t sizes[][2] = {{1, 1}, {7, 5}, {9, 0}, {13, 11}, {4, 4}};
    for (const auto &sz : sizes) {
        int64_t res[2] = {-1, -1};
        run_timed(kernel, sz[0], sz[1], res);
        // the outer body runs once per pass, the inner loop once per pass
        const int64_t passes
                = simulate_unroll(pow2_factors(4), sz[0], 1).passes;
        EXPECT_EQ(res[0], passes * sz[1]) << sz[0] << "x" << sz[1];
        EXPECT_EQ(res[1], sz[0]) << sz[0] << "x" << sz[1];
    }
}

/// A value inside one body iteration may use both loops' lent `tmp`s
TEST(loop_lending, inside_body_value_survives_the_reloads) {
    int r_id = -1;
    buffer::buffer_t lent;
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        const Reg n_outer = a0, n_inner = a1, out = a2;
        const Reg total = a3;
        env_t env(gen, excluding_ints({n_outer, n_inner, out, total}));

        gen.mv(total, x0);

        loop::loop_t *outer = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_outer), /*max_ur=*/4));
        loop::loop_t *inner = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_inner), /*max_ur=*/4));
        optimization::group_t nest(optimization::order_t::level_first);
        nest.declare(*outer, *inner);
        nest.set_level(2);

        Reg r;
        outer->allocate();
        env.pool.scope(outer->released(), [&] {
            inner->allocate();
            return env.pool.scope(inner->released(), [&] {
                r = Reg(env.pool.integers().pop());
                return shortfall_t {};
            });
        });
        r_id = reg_name(r);
        lent = outer->released() | inner->released();

        outer->generate([&](int) {
            inner->generate([&](int ib) {
                gen.li(r, ib);
                gen.add(total, total, r);
                gen.li(r, -1); // clobber before the loop reloads its tmp
            });
        });
        gen.sd(total, out, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_TRUE(lent.has(r_id));

    const int64_t sizes[][2] = {{1, 1}, {7, 5}, {9, 0}, {13, 11}};
    for (const auto &sz : sizes) {
        int64_t res[1] = {-1};
        run_timed(kernel, sz[0], sz[1], res);
        const int64_t passes
                = simulate_unroll(pow2_factors(4), sz[0], 1).passes;
        EXPECT_EQ(res[0], passes * sz[1]) << sz[0] << "x" << sz[1];
    }
}

/// A value spanning a dispatch is clobbered by the loop's lent `tmp` writes
TEST(loop_lending, spanning_value_is_clobbered_by_the_inner_dispatch) {
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        const Reg n_inner = a0, out = a1;
        env_t env(gen, excluding_ints({n_inner, out}));

        loop::loop_t *inner = env.loops.new_loop(loop::plan_t::reduce(
                size_parameter_t::runtime(n_inner), /*max_ur=*/4));
        ASSERT_TRUE(inner->set_level(1));
        ASSERT_TRUE(inner->allocate().ok());
        const Reg r = Reg(inner->released().id(0));

        gen.li(r, 0x55);
        inner->generate([&](int) {});
        gen.sd(r, out, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t res = -1;
    run_timed(kernel, int64_t(3), &res);
    EXPECT_NE(res, 0x55);
}

/// Persistence levels bind a variable to a buffer first, then to the stack
TEST(loop_lending, persistence_trades_a_register_for_a_stack_slot) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());

    auto persist = loop::persistence_t::buffered_then_stack();
    persist.bind(h->frame(), v);
    ASSERT_EQ(persist.levels(), 2);

    const int regs = env.pool.integers().available();
    ASSERT_TRUE(env.pool.begin());
    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(persist.set_level(0));
    EXPECT_EQ(env.pool.integers().available(), regs);
    ASSERT_TRUE(h->allocate().ok());
    const int buffered = regs - env.pool.integers().available();
    ASSERT_TRUE(env.loops.drop());
    ASSERT_TRUE(env.pool.drop());
    EXPECT_EQ(env.pool.integers().available(), regs);
    EXPECT_EQ(h->frame().size(), 0);

    ASSERT_TRUE(env.pool.begin());
    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(persist.set_level(1));
    ASSERT_TRUE(h->allocate().ok());
    const int stacked = regs - env.pool.integers().available();
    EXPECT_EQ(h->frame().size(), 1);
    ASSERT_TRUE(env.loops.commit());
    ASSERT_TRUE(env.pool.commit());

    EXPECT_EQ(buffered, stacked + 1);
    EXPECT_EQ(loop::persistence_t::stacked().levels(), 1);
}

/// A variable without a buffer (an absent feature) is not bound
TEST(loop_lending, persistence_skips_variables_without_buffers) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    const variable_t absent;
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());

    auto persist = loop::persistence_t::buffered_then_stack();
    persist.bind(h->frame(), absent, v, absent);
    ASSERT_TRUE(persist.set_level(1));
    EXPECT_EQ(h->frame().size(), 1);
}

/// Moving a binding to the stack frees a register only for a multi-trip loop
TEST(loop_lending, persistence_relief_follows_the_trips) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    const auto w = env.pool.new_variable(source_t::adopt(a1));
    loop::loop_t *many = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    loop::loop_t *once = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(1), 1));
    ASSERT_TRUE(many->has_multiple_trips());
    ASSERT_FALSE(once->has_multiple_trips());

    auto on_many = loop::persistence_t::buffered_then_stack();
    on_many.bind(*many, v, w);
    EXPECT_EQ(on_many.demand()[resource_t::integers], 2);
    EXPECT_EQ(on_many.relief(0, 1)[resource_t::integers], 2);
    EXPECT_EQ(on_many.relief(0, 1)[resource_t::stack], -2);

    auto on_once = loop::persistence_t::buffered_then_stack();
    on_once.bind(*once, v, w);
    EXPECT_TRUE(on_once.relief(0, 1).is_none());
    EXPECT_TRUE(on_once.relieves().ok());

    loop::frame_t phase;
    auto borrow = loop::persistence_t::on_demand();
    borrow.bind(phase, v, w);
    EXPECT_EQ(borrow.relief(0, 1)[resource_t::integers], 2);
    EXPECT_EQ(borrow.relief(1, 0)[resource_t::integers], 0);
}

/// Loops sharing registers by aliasing stop lending the shared counter
TEST(loop_lending, alias_pair_lends_only_tmp) {
    host_env_t env;
    loop::loop_t *narrow = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::runtime(a0), /*max_ur=*/2));
    ASSERT_TRUE(narrow->is_resolved());
    loop::loop_t *main = env.loops.alias(
            loop::plan_t::reduce(size_parameter_t::runtime(a0), 8), *narrow);
    ASSERT_TRUE(main->is_resolved());
    EXPECT_EQ(narrow->levels(), 2);
    EXPECT_EQ(main->levels(), 2);
    ASSERT_TRUE(main->set_level(1));
    ASSERT_TRUE(narrow->allocate().ok());
    ASSERT_TRUE(main->allocate().ok());
    EXPECT_EQ(main->released().size(), 1);
}

} // namespace rvjit
