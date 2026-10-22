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

// loop::frame_t

/// A `multi_trip` binding on a single-trip loop is inert and allocates nothing
TEST(loop_binding, multi_trip_is_inert_on_a_single_trip_loop) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(1), 8));
    ASSERT_TRUE(h->is_resolved());
    ASSERT_FALSE(h->has_multiple_trips());

    const int free_regs = env.pool.integers().available();
    ASSERT_TRUE(h->frame().bind(v, loop::bind_t::buffer().multi_trip()));
    ASSERT_TRUE(h->allocate().ok());
    EXPECT_EQ(h->frame().size(), 1);
    EXPECT_FALSE(h->frame().has_active());
    EXPECT_EQ(env.pool.integers().available(), free_regs);
}

/// Kind factories plan like `plan_t`; `alias` takes its source's kind
TEST(loop_binding, kind_factories_and_alias) {
    host_env_t env;
    loop::loop_t *r = env.loops.reduce(size_parameter_t::runtime(a0), 2);
    ASSERT_TRUE(r->is_resolved());
    EXPECT_EQ(r->kind(), loop::plan_t::kind_t::reduce);

    loop::loop_t *a = env.loops.alias(*r, size_parameter_t::runtime(a0), 8);
    ASSERT_TRUE(a->is_resolved());
    EXPECT_EQ(a->kind(), loop::plan_t::kind_t::reduce);

    loop::loop_t *d = env.loops.data_parallel(size_parameter_t::literal(4), 1);
    ASSERT_TRUE(d->is_resolved());
    EXPECT_EQ(d->kind(), loop::plan_t::kind_t::data_parallel);
}

/// Bodies reaching a 4 KiB branch need the medium distance
TEST(loop_binding, reach_follows_the_body_size) {
    host_env_t env;
    EXPECT_EQ(env.loops.reach(100), control_flow::distance_t::short_);
    EXPECT_EQ(env.loops.reach(1024), control_flow::distance_t::medium);
}

/// Only a loop counting its trips down names a remaining register
TEST(loop_binding, remaining_register_follows_the_trip_count) {
    host_env_t env;
    loop::loop_t *single = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(1), 8));
    loop::loop_t *many = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::runtime(a0), 2));
    ASSERT_TRUE(single->allocate().ok());
    ASSERT_TRUE(many->allocate().ok());
    EXPECT_FALSE(single->has_remaining());
    EXPECT_TRUE(many->has_remaining());
}

/// The same variable bound to a multi-trip loop gets its slot
TEST(loop_binding, multi_trip_is_active_on_a_multi_trip_loop) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());
    ASSERT_TRUE(h->has_multiple_trips());

    const int free_regs = env.pool.integers().available();
    ASSERT_TRUE(h->frame().bind(v, loop::bind_t::buffer().multi_trip()));
    EXPECT_EQ(env.pool.integers().available(), free_regs);
    ASSERT_TRUE(h->frame().allocate(env.pool, env.stack, true).ok());
    EXPECT_TRUE(h->frame().has_active());
    EXPECT_EQ(env.pool.integers().available(), free_regs - 1);
}

/// Binding a variable twice to one loop replaces its binding
TEST(loop_binding, rebinding_replaces) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());

    ASSERT_TRUE(h->frame().bind(v, loop::bind_t::stack()));
    ASSERT_TRUE(h->frame().bind(v, loop::bind_t::given(slot_t(a5))));
    EXPECT_EQ(h->frame().size(), 1);
}

/// Bindings of one register on different loops are independent
TEST(loop_binding, same_variable_binds_per_loop) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *outer = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    loop::loop_t *inner = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(outer->is_resolved());
    ASSERT_TRUE(inner->is_resolved());

    ASSERT_TRUE(outer->frame().bind(v, loop::bind_t::stack()));
    EXPECT_EQ(outer->frame().size(), 1);
    EXPECT_EQ(inner->frame().size(), 0);
    ASSERT_TRUE(inner->frame().bind(v, loop::bind_t::buffer()));
    EXPECT_EQ(inner->frame().size(), 1);
}

/// A step can only be set on a bound variable
TEST(loop_binding, set_step_requires_a_binding) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a0));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());

    EXPECT_FALSE(h->frame().set_step(v, step_t::plain(const_t(1))));
    ASSERT_TRUE(h->frame().bind(v, loop::bind_t::stack()));
    EXPECT_TRUE(h->frame().set_step(v, step_t::plain(const_t(1))));
}

/// A dropped transaction removes the bindings made within it
TEST(loop_binding, drop_removes_and_commit_keeps) {
    host_env_t env;
    const auto kept = env.pool.new_variable(source_t::adopt(a0));
    const auto dropped = env.pool.new_variable(source_t::adopt(a1));
    const auto committed = env.pool.new_variable(source_t::adopt(a2));
    loop::loop_t *h = env.loops.new_loop(
            loop::plan_t::reduce(size_parameter_t::literal(4), 1));
    ASSERT_TRUE(h->is_resolved());
    ASSERT_TRUE(h->frame().bind(kept, loop::bind_t::stack()));

    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(h->frame().bind(dropped, loop::bind_t::stack()));
    ASSERT_TRUE(env.loops.drop());
    EXPECT_FALSE(h->frame().set_step(dropped, step_t::plain(const_t(1))));
    EXPECT_TRUE(h->frame().set_step(kept, step_t::plain(const_t(1))));

    ASSERT_TRUE(env.loops.begin());
    ASSERT_TRUE(h->frame().bind(committed, loop::bind_t::stack()));
    ASSERT_TRUE(env.loops.commit());
    EXPECT_TRUE(h->frame().set_step(committed, step_t::plain(const_t(1))));
}

/// Loop storage is bounded, and a failed loop never consumes a slot
TEST(loop_binding, new_loop_fails_when_storage_is_exhausted) {
    host_env_t env;
    const auto plan = loop::plan_t::reduce(size_parameter_t::literal(1), 8);
    for (int i = 0; i < rv64::loop::loopgen_t::CAPACITY; ++i)
        ASSERT_TRUE(env.loops.new_loop(plan)->is_resolved());
    EXPECT_FALSE(env.loops.new_loop(plan)->is_resolved());
}

/// An alias source must be a loop of the same component
TEST(loop_binding, alias_rejects_a_foreign_loop) {
    host_env_t env, other;
    const auto plan = loop::plan_t::reduce(size_parameter_t::literal(1), 8);
    loop::loop_t *foreign = other.loops.new_loop(plan);
    ASSERT_TRUE(foreign->is_resolved());
    EXPECT_FALSE(env.loops.alias(plan, *foreign)->is_resolved());
}

/// An inert binding emits nothing, so a single-trip loop's value never moves
TEST(loop_binding, inert_binding_does_not_step) {
    static const int64_t start = 10;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        const auto entry = env.pool.new_variable(source_t::adopt(r));
        loop::loop_t *desc = env.loops.new_loop(
                loop::plan_t::reduce(size_parameter_t::literal(1), 8));
        desc->frame().bind(entry,
                loop::bind_t::reloaded().multi_trip().stepped(
                        step_t::plain(const_t(7))));
        desc->allocate();
        desc->generate([&](int) {});

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, &out);
    EXPECT_EQ(out, start);
}

/// A standalone frame saves its variables around a phase that clobbers them
TEST(loop_frame, standalone_frame_restores_around_a_phase) {
    static const int64_t start = 5;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        loop::frame_t phase;
        const auto v = env.pool.new_variable(source_t::adopt(r));
        ASSERT_TRUE(phase.bind(v, loop::bind_t::stack()));
        {
            const shortfall_t s = phase.allocate(env.pool, env.stack, true);
            ASSERT_TRUE(s.ok());
        }
        EXPECT_EQ(phase.restored(v), buffer::buffer_t({r}));

        env.stack.reserve_stack();
        phase(env.xfer, [&] { gen.li(r, -1); });
        env.stack.release_stack();

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    run_timed(kernel, &out);
    EXPECT_EQ(out, start);
}

/// restored() keeps the listed variables reloaded after the body
TEST(loop_frame, restored_filters_inactive_slotless_and_unlisted) {
    host_env_t env;
    const auto stacked = env.pool.new_variable(source_t::adopt(a1));
    const auto off = env.pool.new_variable(source_t::adopt(a2));
    const auto single = env.pool.new_variable(source_t::adopt(a3));
    const auto slotless = env.pool.new_variable(source_t::adopt(a4));
    const auto copied = env.pool.new_variable(source_t::copy(a5));
    const auto unbound = env.pool.new_variable(source_t::adopt(a6));

    loop::frame_t f;
    ASSERT_TRUE(f.bind(stacked, loop::bind_t::stack()));
    ASSERT_TRUE(f.bind(off, loop::bind_t::off()));
    ASSERT_TRUE(f.bind(single, loop::bind_t::stack().multi_trip()));
    ASSERT_TRUE(f.bind(slotless, loop::bind_t::reloaded()));
    ASSERT_TRUE(f.bind(copied, loop::bind_t::reloaded()));
    ASSERT_TRUE(f.allocate(env.pool, env.stack, /*multi_trip=*/false).ok());

    const buffer::buffer_t regs
            = f.restored(stacked, off, single, slotless, copied, unbound);
    EXPECT_EQ(regs, buffer::buffer_t({stacked.buffer(), copied.buffer()}));
}

/// On-demand persistence is unbound at level 0 and stack-saved at level 1
TEST(loop_frame, on_demand_binds_only_when_relaxed) {
    host_env_t env;
    const auto v = env.pool.new_variable(source_t::adopt(a1));
    loop::frame_t f;
    auto borrow = loop::persistence_t::on_demand();
    borrow.bind(f, v);
    ASSERT_EQ(borrow.levels(), 2);

    ASSERT_TRUE(borrow.set_level(0));
    ASSERT_TRUE(f.allocate(env.pool, env.stack, true).ok());
    EXPECT_FALSE(f.has_active());
    EXPECT_TRUE(f.restored(v).is_empty());

    ASSERT_TRUE(borrow.set_level(1));
    ASSERT_TRUE(f.allocate(env.pool, env.stack, true).ok());
    EXPECT_TRUE(f.has_active());
    EXPECT_EQ(f.restored(v), buffer::buffer_t({a1}));
}

} // namespace rvjit
