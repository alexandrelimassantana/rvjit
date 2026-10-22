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
#include "rv64/stream/iterator/iterator_helpers.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;

// Host-side: plan()'s resolution logic (no codegen involved)

/// A direct axis costs 0 registers at axis 0, at size 1, or walking the
/// advancing origin, else 1
TEST(iterator_plan_resolution, direct_axis_register_cost) {
    // 1D forward: 0 registers regardless of size
    {
        space_t space;
        space.axis(5, const_t(8));
        auto plan
                = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                                rv64::stream::iterator::guided),
                        0);
        ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(0).ngpr, 0);
    }
    // 1D preserve: 1 register when size > 1
    {
        space_t space;
        space.axis(5, const_t(8));
        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(a3, 0), space,
                        rv64::stream::iterator::guided, addressing_t::base_simm,
                        update_t::preserve()),
                1);
        ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(0).ngpr, 1);
    }
    // 2D direct/direct: dim 1 needs a register iff its own size > 1
    {
        space_t space;
        space.axis(3, const_t(64)).axis(1, const_t(8));
        auto plan
                = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                                rv64::stream::iterator::guided),
                        0);
        ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(0).ngpr, 0);
        EXPECT_EQ(plan.axis(1).ngpr, 0) << "N x 1 tile needs no column reg";
    }
    // Dim 1 walks the origin when its rewind folds into the row step
    {
        space_t space;
        space.axis(3, const_t(64)).axis(4, const_t(8));
        auto plan
                = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                                rv64::stream::iterator::guided),
                        0);
        ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(1).ngpr, 0);
        EXPECT_TRUE(plan.axis(1).takes_origin);
    }
    // A register row stride leaves no step to fold into
    {
        space_t space;
        space.axis(3, const_t(a4)).axis(4, const_t(8));
        auto plan
                = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                                rv64::stream::iterator::guided),
                        1);
        ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(1).ngpr, 1);
        EXPECT_FALSE(plan.axis(1).takes_origin);
    }
    // A zero row step is elided, leaving the rewind nothing to fold into
    {
        space_t space;
        space.axis(1, const_t(0)).axis(2, const_t(256));
        auto plan
                = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                                rv64::stream::iterator::guided),
                        1);
        ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(1).ngpr, 1);
        EXPECT_FALSE(plan.axis(1).takes_origin);
    }
    // A preserving walk never takes its origin
    {
        space_t space;
        space.axis(3, const_t(64)).axis(4, const_t(8));
        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(a3, 0), space,
                        rv64::stream::iterator::guided, addressing_t::base_simm,
                        update_t::preserve()),
                2);
        ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
        EXPECT_EQ(plan.axis(1).ngpr, 1);
        EXPECT_FALSE(plan.axis(1).takes_origin);
    }
}

/// A pivot axis whose full span fits simm12 (case a) takes 0 registers
TEST(iterator_plan_resolution, pivot_case_a) {
    // (size-1)*stride fits simm12 directly: 0 registers
    space_t space;
    space.axis(64, const_t(8)); // span = 63*8 = 504
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            0);
    ASSERT_TRUE(plan.is_valid()); // 0 registers needed: already fully resolved
    EXPECT_EQ(plan.axis(0).ngpr, 0);
}

/// A pivot axis fitting one shifted window (case b) takes exactly 1 register,
/// the advancing origin's
TEST(iterator_plan_resolution, pivot_case_b) {
    // span past 2047 but within one shifted window: the origin is the pivot
    space_t space;
    space.axis(300, const_t(8)); // span=299*8=2392
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            1);
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(0).ngpr, 0);
    EXPECT_TRUE(plan.axis(0).takes_origin);
    EXPECT_EQ(plan.axis(0).shift, 2047);
    EXPECT_EQ(plan.origin_bias(), 2047);

    space_t nspace;
    nspace.axis(300, const_t(-8));
    auto nplan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), nspace,
                                               rv64::stream::iterator::pivot),
            1);
    ASSERT_NE(nplan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(nplan.axis(0).ngpr, 0);
    EXPECT_EQ(nplan.axis(0).shift, -2047);

    // A preserving walk draws its pivot
    auto pplan = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space, rv64::stream::iterator::pivot,
                    addressing_t::base_simm, update_t::preserve()),
            1);
    ASSERT_NE(pplan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(pplan.axis(0).ngpr, 1);
    EXPECT_FALSE(pplan.axis(0).takes_origin);
    EXPECT_EQ(pplan.origin_bias(), 0);
}

/// A pivot axis spanning several windows (case c) takes one register per strip
TEST(iterator_plan_resolution, pivot_case_c_multi_strip) {
    // span well past one shifted window: multiple registers
    space_t space;
    space.axis(2000, const_t(8)); // span=1999*8=15992
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            16);
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    // positions_per_pivot=511 (tail-adjusted), so ceil(2000/511) == 4 strips,
    // the first on the advancing origin
    EXPECT_EQ(plan.axis(0).ngpr, 3);
    EXPECT_EQ(plan.axis(0).nregs(), 4);
}

/// Small-span axes sharing one combined immediate all resolve to 0 registers
TEST(iterator_plan_resolution, pivot_shared_offset_group) {
    // Two small-span axes that should both resolve to 0 registers, sharing
    // one combined immediate.
    space_t space;
    space.axis(4, const_t(64)) // worst 3*64=192
            .axis(8, const_t(4)); // worst 7*4=28
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space), 0);
    ASSERT_TRUE(plan.is_valid());
    EXPECT_EQ(plan.axis(0).ngpr, 0);
    EXPECT_EQ(plan.axis(1).ngpr, 0);
}

/// When combined worst cases overflow simm12, the second axis owns registers
TEST(iterator_plan_resolution,
        pivot_shared_offset_group_overflow_bumps_second_axis) {
    // Each axis alone fits simm12, but their combined worst case doesn't;
    // plan resolves greedily left-to-right, so the SECOND axis in
    // declaration order is the one that gets bumped to owning registers.
    space_t space;
    space.axis(200, const_t(8)) // worst 199*8=1592
            .axis(200, const_t(8)); // also 1592 alone
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            4);
    ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(0).ngpr, 0);
    EXPECT_GT(plan.axis(1).ngpr, 0);
}

/// Two axes needing multiple registers beyond the pool budget fail to resolve
TEST(iterator_plan_resolution, multi_register_pivot_restricted_to_one_axis) {
    space_t space;
    space.axis(2000, const_t(8)).axis(2000, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            32);
    EXPECT_FALSE(plan.is_valid());
}

/// `any` tries `pivot` first and falls back to `direct` only if it won't fit
TEST(iterator_plan_resolution, any_prefers_pivot_falls_back_to_direct) {
    // Small span: pivot fits with 0 registers, any picks it
    {
        space_t space;
        space.axis(64, const_t(8));
        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(a3, 0), space), 0);
        ASSERT_TRUE(plan.is_valid());
        EXPECT_EQ(plan.axis(0).strategy, rv64::stream::iterator::pivot);
    }
    // Register-starved: pivot needs multiple registers and none are
    // available, any falls back to direct instead of failing
    {
        space_t space;
        space.axis(2000, const_t(8));
        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(a3, 0), space), 0);
        ASSERT_TRUE(plan.is_valid());
        EXPECT_EQ(plan.axis(0).strategy, rv64::stream::iterator::guided);
        EXPECT_EQ(plan.axis(0).ngpr, 0);
    }
}

/// A case-a pivot end axis at axis 0 steps eagerly, walk() compensating reads
TEST(iterator_plan_resolution, pivot_eager_case_a_offset_sequence) {
    space_t space;
    // Axis 0: case-a (span = 15*4 = 60, well within simm12), the
    // designated (advance-kind) end axis
    // Axis 1: register-sourced stride -- forces `guided` (pivot can't
    // chain a runtime-only stride), matching the draft's own "dynamic"
    space.axis(16, const_t(4)).axis(6, const_t(a4));
    auto plan = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space, strategy_id_t::any(),
                    addressing_t::base_simm, update_t::advance(0)),
            1);
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(0).strategy, rv64::stream::iterator::pivot);
    EXPECT_EQ(plan.axis(0).ngpr, 0); // case-a: 0 registers
    EXPECT_EQ(plan.axis(1).strategy, rv64::stream::iterator::guided);

    static const int expected[16] = {0, -60, -56, -52, -48, -44, -40, -36, -32,
            -28, -24, -20, -16, -12, -8, -4};

    const rv64::stream::iterator::pivot_t pivot;
    for (int i = 0; i < 16; ++i) {
        pointer_t out = plan.origin;
        const index_t idx {i, 0};
        pivot.walk(out, plan, 0, idx);
        EXPECT_EQ(out.offset, expected[i]) << "at i=" << i;
    }
}

// Host-side: addressing_t

/// plan() defaults to immediate addressing
TEST(iterator_stream_addressing, default_addressing_is_immediate) {
    space_t space;
    space.axis(64, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            0);
    EXPECT_EQ(plan.addressing, addressing_t::base_simm);
}

/// `direct` addressing forces one register per position, even for case-a axes
TEST(iterator_stream_addressing,
        direct_addressing_forces_register_per_position_for_immediate_stride) {
    // Under `immediate` addressing this axis resolves to case (a): 0
    // registers (matches PivotCaseA above). Under `direct`, no immediate
    // offset is usable at all, so it must resolve to one register per
    // position instead.
    space_t space;
    space.axis(64, const_t(8));
    auto plan = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space, rv64::stream::iterator::pivot,
                    addressing_t::base_only),
            64);
    // Registers aren't attached by plan() itself (see PivotCaseB/C
    // above), so a resolved-but-unbacked plan isn't is_valid() yet; the
    // advancing origin is the first position's
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(0).ngpr, 63);
    EXPECT_TRUE(plan.axis(0).takes_origin);
}

/// A register-sourced stride takes a pivot register per position, the first
/// being the advancing origin, when they fit, else only guided resolves it
TEST(iterator_stream_addressing,
        register_sourced_stride_pivots_per_position_or_falls_to_guided) {
    space_t space_regstride;
    space_regstride.axis(64, const_t(a4));
    auto plan_regstride_pivot = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space_regstride,
                    rv64::stream::iterator::pivot),
            63);
    ASSERT_EQ(plan_regstride_pivot.axis(0).strategy,
            rv64::stream::iterator::pivot);
    EXPECT_EQ(plan_regstride_pivot.axis(0).ngpr, 63);

    auto plan_regstride_short = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space_regstride,
                    rv64::stream::iterator::pivot),
            62);
    EXPECT_EQ(plan_regstride_short.axis(0).strategy, strategy_id_t::any());

    auto plan_regstride_guided = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space_regstride,
                    rv64::stream::iterator::guided),
            62);
    ASSERT_NE(plan_regstride_guided.axis(0).strategy, strategy_id_t::any());
    {
        const auto strategy = plan_regstride_guided.axis(0).strategy;
        EXPECT_EQ(strategy, rv64::stream::iterator::guided);
    }

    space_t space_direct;
    space_direct.axis(64, const_t(8));
    auto plan_direct = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space_direct,
                    rv64::stream::iterator::pivot, addressing_t::base_only),
            63);
    ASSERT_NE(plan_direct.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(plan_direct.axis(0).ngpr, 63);
}

// Host-side: no fallback for an explicit strategy request

/// An explicit `pivot` that doesn't fit `ntmp` fails, with no fallback
TEST(iterator_pivot_no_fallback,
        explicit_pivot_fails_outright_when_registers_insufficient) {
    space_t space;
    space.axis(2000, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::pivot),
            3);
    EXPECT_FALSE(plan.is_valid());
}

/// An explicit `direct` request that doesn't fit `ntmp` also fails outright
TEST(iterator_pivot_no_fallback,
        explicit_direct_fails_outright_when_registers_insufficient) {
    space_t space;
    space.axis(3, const_t(a4)).axis(4, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                              rv64::stream::iterator::guided),
            0);
    EXPECT_FALSE(plan.is_valid());
}

// Host-side: compositions an axis's own registers cannot follow

/// Moving outer axes reject a pivot register chain, which `any` walks guided
TEST(iterator_plan_composition, register_chain_under_moving_outer_axis) {
    space_t strips;
    strips.axis(8, const_t(1024)).axis(4, const_t(a4));
    const auto pivot = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), strips,
                    rv64::stream::iterator::pivot, addressing_t::base_simm,
                    update_t::preserve()),
            16);
    EXPECT_EQ(pivot.axis(1).strategy, strategy_id_t::any());

    const auto any = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), strips, strategy_id_t::any(),
                    addressing_t::base_simm, update_t::preserve()),
            16);
    ASSERT_NE(any.axis(1).strategy, strategy_id_t::any());
    EXPECT_EQ(any.axis(0).strategy, rv64::stream::iterator::pivot);
    EXPECT_EQ(any.axis(1).strategy, rv64::stream::iterator::guided);

    // Case-a rows add offsets only, so the columns keep their registers
    space_t offsets;
    offsets.axis(8, const_t(16)).axis(4, const_t(a4));
    const auto chain = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), offsets,
                    rv64::stream::iterator::pivot, addressing_t::base_simm,
                    update_t::preserve()),
            16);
    ASSERT_EQ(chain.axis(1).strategy, rv64::stream::iterator::pivot);
    EXPECT_EQ(chain.axis(1).nregs(), 4);
}

/// A guided inner end axis cannot advance under a moving outer axis
TEST(iterator_plan_composition, guided_inner_end_axis_under_moving_outer) {
    space_t rows;
    rows.axis(8, const_t(16)).axis(4, const_t(4));
    const auto inner = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), rows, rv64::stream::iterator::guided,
                    addressing_t::base_simm, update_t::advance(1)),
            16);
    EXPECT_EQ(inner.axis(1).strategy, strategy_id_t::any());

    space_t row;
    row.axis(1, const_t(16)).axis(4, const_t(4));
    const auto single = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), row, rv64::stream::iterator::guided,
                    addressing_t::base_simm, update_t::advance(1)),
            16);
    EXPECT_EQ(single.axis(1).strategy, rv64::stream::iterator::guided);
}

// Multi-axis one-register-per-position pivot chains (axis_plan_t::
// chain_start): a run of 2+ consecutive axes forced onto the
// one-register-per-position path shares one combined register array,
// rather than each axis resolving (and going stale against the others)
// independently -- see plan's own chain-detection pass

/// A base_only 2-axis run from axis 0 resolves as one chain with a delegate
TEST(iterator_plan_resolution, multibase_chain_combines_register_cost) {
    space_t space;
    space.axis(2, const_t(8)).axis(2, const_t(4));
    auto plan = rv64_iterator_t::plan(
            make_request(pointer_t(a3, 0), space, rv64::stream::iterator::pivot,
                    addressing_t::base_only, update_t::advance(0)),
            8);
    // plan never allocates registers itself (that's
    // new_iterator's job) -- a resolved axis needing real registers stays
    // needs_regs()==true (and so is_valid()==false) until then, matching
    // every other pivot-resolution test in this file (e.g. PivotCaseB above)
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(0).ngpr, 0);
    EXPECT_EQ(plan.axis(0).chain_start, 0);
    ASSERT_NE(plan.axis(1).strategy, strategy_id_t::any());
    EXPECT_EQ(plan.axis(1).ngpr, 4);
    EXPECT_EQ(plan.axis(1).chain_start, 0);
}

// Host-side: the registered stream table

namespace {

/// A one-axis plan of `n` positions with its registers allocated
plan_t table_plan(model_env_t &env, int n) {
    space_t space;
    space.axis(n, const_t(8));
    auto p = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                           rv64::stream::iterator::guided),
            1);
    env.iterators.new_iterator(p);
    return p;
}

} // namespace

/// resize() narrows and widens a registered iterator within its registered shape
TEST(iterator_stream_table, resize_stays_within_the_registered_shape) {
    host_model_env_t env;
    auto *h = env.iterators.iterator(table_plan(env, 6));
    ASSERT_TRUE(h->is_valid());
    EXPECT_EQ(h->size(0), 6);

    EXPECT_TRUE(h->resize({4}));
    EXPECT_EQ(h->size(0), 4);
    EXPECT_TRUE(h->resize({6}));
    EXPECT_EQ(h->size(0), 6);

    EXPECT_FALSE(h->resize({7}));
    EXPECT_EQ(h->size(0), 6);
}

/// An unresolved plan registers nothing
TEST(iterator_stream_table, unresolved_plan_yields_nullptr) {
    host_model_env_t env;
    space_t space;
    space.axis(2, const_t(a4)).axis(3, const_t(8));
    auto p = rv64_iterator_t::plan(make_request(pointer_t(a3, 0), space,
                                           rv64::stream::iterator::guided),
            1);
    ASSERT_TRUE(p.needs_regs());
    EXPECT_FALSE(env.iterators.iterator(p)->is_valid());
}

// strategy_ladder_t

namespace {

/// A one-axis walk of 4 positions `stride` bytes apart, under `mode`
stream::iterator::request_t ladder_request(
        int stride, addressing_t mode = addressing_t::base_simm) {
    space_t space;
    space.axis(4, const_t(stride));
    return make_request(pointer_t(a3, 0), space, strategy_id_t::any(), mode);
}

} // namespace

/// Without a hint the levels follow the set's cost ladder
TEST(strategy_ladder, levels_follow_the_set_order) {
    stream::iterator::strategy_id_t out;
    stream::iterator::strategy_ladder_t ladder(
            rv64::stream::iterator::strategies(), ladder_request(8),
            stream::iterator::strategy_id_t::any(), -1, out);
    ASSERT_EQ(ladder.levels(), 2);
    EXPECT_EQ(out, rv64::stream::iterator::pivot);

    ASSERT_TRUE(ladder.set_level(1));
    EXPECT_EQ(out, rv64::stream::iterator::guided);
    EXPECT_EQ(ladder.level(), 1);
    ASSERT_TRUE(ladder.set_level(0));
    EXPECT_EQ(out, rv64::stream::iterator::pivot);
    EXPECT_FALSE(ladder.set_level(2));
}

/// An explicit hint is the only level
TEST(strategy_ladder, a_hint_is_the_only_level) {
    stream::iterator::strategy_id_t out;
    stream::iterator::strategy_ladder_t ladder(
            rv64::stream::iterator::strategies(), ladder_request(8),
            rv64::stream::iterator::guided, -1, out);
    EXPECT_EQ(ladder.levels(), 1);
    EXPECT_EQ(out, rv64::stream::iterator::guided);
    EXPECT_TRUE(ladder.set_level(0));
    EXPECT_TRUE(ladder.relieves().ok());
}

/// Relief between levels is the registers each strategy saves in the walk
TEST(strategy_ladder, relief_follows_the_registers_each_level_needs) {
    using iterator_t = stream::iterator::iterator_t;
    const auto &set = rv64::stream::iterator::strategies();

    // One register per position for pivot, under base-only addressing
    auto costly = ladder_request(8, addressing_t::base_only);
    stream::iterator::strategy_id_t out;
    stream::iterator::strategy_ladder_t ladder(
            set, costly, strategy_id_t::any(), -1, out);
    costly.strategy = rv64::stream::iterator::pivot;
    const int pivot = iterator_t::need(set, costly, -1);
    costly.strategy = rv64::stream::iterator::guided;
    const int guided = iterator_t::need(set, costly, -1);
    ASSERT_GT(pivot, guided);
    EXPECT_EQ(ladder.relief(0, 1)[resource_t::integers], pivot - guided);
    EXPECT_EQ(ladder.demand()[resource_t::integers], pivot);
    {
        const shortfall_t ints = shortfall_t::of(resource_t::integers);
        EXPECT_TRUE(ladder.relieves().intersects(ints));
    }
    ASSERT_TRUE(ladder.set_level(ladder.levels() - 1));
    EXPECT_TRUE(ladder.relieves().ok());

    // A pivot span fitting one immediate needs no register: nothing to relieve
    stream::iterator::strategy_ladder_t cheap(
            set, ladder_request(8), strategy_id_t::any(), -1, out);
    EXPECT_EQ(cheap.demand()[resource_t::integers], 0);
    EXPECT_TRUE(cheap.relieves().ok());
}

/// A level its cap cannot host costs an unbounded count, never a relief
TEST(strategy_ladder, a_capped_level_is_unbounded) {
    stream::iterator::strategy_id_t out;
    stream::iterator::strategy_ladder_t ladder(
            rv64::stream::iterator::strategies(),
            ladder_request(8, addressing_t::base_only), strategy_id_t::any(),
            /*ntmp=*/1, out);
    EXPECT_EQ(ladder.demand()[resource_t::integers], demand_t::INF);
    EXPECT_EQ(ladder.relief(0, 1)[resource_t::integers], demand_t::INF);
    EXPECT_EQ(ladder.relief(1, 0)[resource_t::integers], -demand_t::INF);
}

} // namespace rvjit
