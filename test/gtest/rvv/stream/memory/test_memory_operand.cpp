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
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvv/stream/memory/memory_operand_helpers.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;
using rv64_iterator_t = rvjit::rv64::stream::iterator::iterator_t;
using memory_operand_t = rvjit::stream::memory_operand::memory_operand_t;
using memory_spec_t = rvjit::stream::memory_operand::spec_t;
using memory_variant_t = rvjit::stream::memory_operand::variant_t;

// Host-side: spec validation and creation failures (no code is executed)

/// A vector-kind memop needs `base_only` addressing, whatever the strategy
TEST(memory_operand_spec, vector_kind_requires_base_only_addressing) {
    space_t space;
    space.axis(64, const_t(8));
    const memory::op_t op = memory::op_t().vector().load().as_type(type_t::f32);

    {
        const auto spec = make_spec(
                op, pointer_t(a0, 0), space, addressing_t::base_simm);
        EXPECT_FALSE(spec.is_valid());
    }
    {
        const auto spec = make_spec(
                op, pointer_t(a0, 0), space, addressing_t::base_only);
        EXPECT_TRUE(spec.is_valid());
    }
}

/// A scalar memop accepts either addressing
TEST(memory_operand_spec, scalar_kind_accepts_both_addressings) {
    space_t space;
    space.axis(4, const_t(8));
    const memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s32);

    {
        const auto spec = make_spec(
                op, pointer_t(a0, 0), space, addressing_t::base_simm);
        EXPECT_TRUE(spec.is_valid());
    }
    {
        const auto spec = make_spec(
                op, pointer_t(a0, 0), space, addressing_t::base_only);
        EXPECT_TRUE(spec.is_valid());
    }
}

/// create() refuses an invalid spec and reports a shortfall
TEST(memory_operand_create, invalid_spec_yields_an_invalid_stream) {
    host_model_env_t env;
    space_t space;
    space.axis(64, const_t(8));
    const memory::op_t op = memory::op_t().vector().load().as_type(type_t::f32);

    auto *s = env.memops.create(
            make_spec(op, pointer_t(t0, 0), space, addressing_t::base_simm),
            make_variant(strategy_id_t::any(), 0, buffer_map_t({2, 1})));
    EXPECT_FALSE(s->is_valid());
    EXPECT_EQ(s, memory_operand_t::invalid());
    EXPECT_FALSE(env.memops.miss().ok());
}

/// A pivot axis under `base_only` needs one register per position, the
/// advancing origin being the first
TEST(memory_operand_create,
        pivot_under_base_only_needs_one_register_per_position) {
    host_model_env_t env;
    space_t space;
    space.axis(4, const_t(8));
    const memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s32);
    auto spec = make_spec(op, pointer_t(t0, 0), space, addressing_t::base_only,
            update_t::advance(0));

    auto *tight = env.memops.create(spec,
            make_variant(
                    rv64::stream::iterator::pivot, 2, buffer_map_t({4, 1})));
    EXPECT_FALSE(tight->is_valid());

    auto *roomy = env.memops.create(spec,
            make_variant(
                    rv64::stream::iterator::pivot, 3, buffer_map_t({4, 1})));
    EXPECT_TRUE(roomy->is_valid());
}

/// create() reports an integer shortfall once no register is left for the ring
TEST(memory_operand_create, pool_exhaustion_reports_the_shortfall) {
    host_model_env_t env;
    space_t space;
    space.axis(4, const_t(8));
    const memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s32);
    while (env.pool.integers().available() > 0)
        Reg(env.pool.integers().pop());

    auto *s = env.memops.create(make_spec(op, pointer_t(t0, 0), space),
            make_variant(
                    rv64::stream::iterator::guided, 0, buffer_map_t({2, 1})));
    EXPECT_FALSE(s->is_valid());
    {
        const shortfall_t lack = shortfall_t::of(resource_t::integers);
        EXPECT_TRUE(env.memops.miss().intersects(lack));
    }
}

/// create() draws the ring from the pool when the variant brings none
TEST(memory_operand_create, resolves_pointers_and_ring) {
    host_model_env_t env;
    space_t space;
    space.axis(4, const_t(8));
    const memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s32);

    auto *s = env.memops.create(
            make_spec(op, pointer_t(t0, 0), space, addressing_t::base_simm,
                    update_t::preserve()),
            make_variant(rv64::stream::iterator::guided,
                    env.pool.integers().available(), buffer_map_t({2, 1})));
    ASSERT_TRUE(s->is_valid());
    EXPECT_EQ(s->buffer().size(), 2);
    EXPECT_TRUE(env.memops.miss().ok());
}

/// replan() re-resolves the stream with another strategy, keeping its spec
TEST(memory_operand_create, replan_switches_the_strategy) {
    host_model_env_t env;
    space_t space;
    space.axis(4, const_t(8));
    const memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

    auto *s = env.memops.create(make_spec(op, pointer_t(t0, 0), space),
            make_variant(rv64::stream::iterator::pivot,
                    env.pool.integers().available(), buffer_map_t({2, 1})));
    ASSERT_TRUE(s->is_valid());
    {
        const auto v = make_variant(rv64::stream::iterator::guided,
                env.pool.integers().available(), buffer_map_t({2, 1}));
        ASSERT_TRUE(s->replan(v));
    }
    EXPECT_TRUE(s->is_valid());
    EXPECT_EQ(s->size(0), 4);
}

// plan_t: the levers a resource search steps a memory operand through

using memory_plan_t = rvjit::stream::memory_operand::plan_t;
using memory_axis_role_t = rvjit::stream::memory_operand::axis_role_t;

memory_spec_t make_roles(
        type_t dt, memory_axis_role_t k, memory_axis_role_t n) {
    space_t space;
    space.axis(4, const_t(8));
    space.axis(4, const_t(32));
    memory_spec_t spec = make_spec(memory::op_t().scalar().load().as_type(dt),
            pointer_t(t0, 0), space);
    spec.role[0] = k;
    spec.role[1] = n;
    return spec;
}

/// An integer buffer has one lever: reduce axes, strategy, then parallel axes
TEST(memory_operand_plan,
        integer_buffer_relaxes_reduce_then_strategy_then_parallel) {
    host_model_env_t env;
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::s32, memory_axis_role_t::reduce,
                    memory_axis_role_t::parallel),
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    auto levers = plan.levers();
    ASSERT_EQ(levers.size(), 1);
    ASSERT_EQ(levers[0]->levels(), 5);

    const int rings[5][2] = {{2, 3}, {1, 3}, {1, 3}, {1, 2}, {1, 1}};
    for (int l = 0; l < 5; ++l) {
        ASSERT_TRUE(levers[0]->set_level(l));
        const auto v = plan.variant();
        EXPECT_EQ(v.ring.shape[0], rings[l][0]) << "level " << l;
        EXPECT_EQ(v.ring.shape[1], rings[l][1]) << "level " << l;
        {
            const auto want = l < 2 ? rv64::stream::iterator::pivot
                                    : rv64::stream::iterator::guided;
            EXPECT_EQ(v.strategy, want) << "level " << l;
        }
    }
}

/// A float buffer has a ring lever for floats and a walk lever for integers
TEST(memory_operand_plan, float_buffer_splits_the_ring_and_the_walk) {
    host_model_env_t env;
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::f32, memory_axis_role_t::reduce,
                    memory_axis_role_t::parallel),
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    auto levers = plan.levers();
    ASSERT_EQ(levers.size(), 2);
    EXPECT_EQ(levers[0]->levels(), 4);
    {
        const shortfall_t lack = shortfall_t::of(resource_t::floats);
        EXPECT_TRUE(levers[0]->relieves().intersects(lack));
    }
    {
        const shortfall_t lack = shortfall_t::of(resource_t::integers);
        EXPECT_FALSE(levers[0]->relieves().intersects(lack));
    }
    EXPECT_EQ(levers[1]->levels(), 2);

    // Both spans fit an immediate: the pivot walk needs no register, so the
    // cheaper strategy relieves nothing
    EXPECT_EQ(plan.demand()[resource_t::integers], 0);
    EXPECT_LE(levers[1]->relief(0, 1)[resource_t::integers], 0);
    EXPECT_TRUE(levers[1]->relieves().ok());

    ASSERT_TRUE(levers[0]->set_level(levers[0]->levels() - 1));
    EXPECT_TRUE(levers[0]->relieves().ok());
}

/// Lever relief equals the plan's demand difference, ring levers sharing a ring
TEST(memory_operand_plan, lever_relief_is_the_demand_difference) {
    host_model_env_t env;
    memory_spec_t spec = make_roles(type_t::s32, memory_axis_role_t::reduce,
            memory_axis_role_t::parallel);
    spec.addressing = rvjit::stream::iterator::addressing_t::base_only;
    memory_plan_t plan(env.memops.strategies(), spec,
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    resource_optimizable_t &lever = *plan.levers()[0];
    std::vector<demand_t> at;
    for (int l = 0; l < lever.levels(); ++l) {
        ASSERT_TRUE(lever.set_level(l));
        at.push_back(plan.demand());
    }
    ASSERT_TRUE(lever.set_level(0));
    EXPECT_EQ(at[0][resource_t::integers], 6 + 16); // ring, one per position
    for (int a = 0; a < lever.levels(); ++a)
        for (int b = 0; b < lever.levels(); ++b) {
            const auto got = lever.relief(a, b)[resource_t::integers];
            const auto want
                    = at[a][resource_t::integers] - at[b][resource_t::integers];
            EXPECT_EQ(got, want) << a << " -> " << b;
        }
}

/// A wrapped ring shrinks its depth as axis 0, and only when that is `reduce`
TEST(memory_operand_plan, wrapped_ring_shrinks_its_depth) {
    host_model_env_t env;
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::s32, memory_axis_role_t::reduce,
                    memory_axis_role_t::parallel),
            make_variant(
                    strategy_id_t::any(), -1, buffer_map_t({4, 2}).wrapped(3)));

    auto levers = plan.levers();
    ASSERT_EQ(levers.size(), 1);
    ASSERT_EQ(levers[0]->levels(), 4);
    for (int l = 0; l < 3; ++l) {
        ASSERT_TRUE(levers[0]->set_level(l));
        const auto v = plan.variant();
        EXPECT_TRUE(v.ring.is_wrapped());
        EXPECT_EQ(v.ring.total(), 3 - l) << "level " << l;
        EXPECT_EQ(v.ring.shape[0], 4);
        EXPECT_EQ(v.ring.shape[1], 2);
    }

    memory_plan_t fixed(env.memops.strategies(),
            make_roles(type_t::s32, memory_axis_role_t::fixed,
                    memory_axis_role_t::reduce),
            make_variant(
                    strategy_id_t::any(), -1, buffer_map_t({4, 2}).wrapped(3)));
    EXPECT_EQ(fixed.levers()[0]->levels(), 2);
}

/// A fixed axis is never shrunk
TEST(memory_operand_plan, fixed_axes_keep_their_extent) {
    host_model_env_t env;
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::s32, memory_axis_role_t::reduce,
                    memory_axis_role_t::fixed),
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    auto levers = plan.levers();
    ASSERT_EQ(levers[0]->levels(), 3);
    ASSERT_TRUE(levers[0]->set_level(2));
    EXPECT_EQ(plan.variant().ring.shape[1], 3);
}

/// The ring is allocated first, so a float ring leaves the walk every integer
TEST(memory_operand_plan, float_ring_leaves_the_integers_to_the_pointers) {
    host_model_env_t env;
    const int ints = env.pool.integers().available();
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::f32, memory_axis_role_t::reduce,
                    memory_axis_role_t::parallel),
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    auto *s = env.memops.create(plan);
    ASSERT_TRUE(s->is_valid());
    EXPECT_EQ(s->buffer().size(), 6);
    EXPECT_GE(env.pool.integers().available(), ints - 4);
}

/// An exhausted float file is reported as a float shortfall
TEST(memory_operand_plan, float_ring_shortfall_names_floats) {
    host_model_env_t env;
    while (env.pool.floats().available() > 0)
        FReg(env.pool.floats().pop());
    memory_plan_t plan(env.memops.strategies(),
            make_roles(type_t::f32, memory_axis_role_t::reduce,
                    memory_axis_role_t::parallel),
            make_variant(strategy_id_t::any(), -1, buffer_map_t({2, 3})));

    EXPECT_FALSE(env.memops.create(plan)->is_valid());
    {
        const shortfall_t lack = shortfall_t::of(resource_t::floats);
        EXPECT_TRUE(env.memops.miss().intersects(lack));
    }
}

} // namespace rvjit
