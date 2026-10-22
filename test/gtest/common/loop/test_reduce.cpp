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

#include "gtest/gtest.h"

#include "loop_mock.hpp"
#include "rvjit/implementation/common/loop/reduce.hpp"
#include "stream_mock.hpp"

namespace rvjit {

/// reduce_t interleaves the inits once, then narrows rs1/rs2 per K stage
TEST(reduce_loop, resizes_rs1_rs2_per_stage) {
    common::stream::combinator_t combinator;

    loop::reduce_t reduce(combinator);

    int rd_init_calls = 0, rs1_init_calls = 0, rs2_init_calls = 0;

    // rd: {M, N} = {2, 2}; rs1 ("B"): {K, N} = {3, 2}; rs2 ("A"): {K, M} = {3, 2}
    mock_streams_t streams;
    auto rd = streams.add(
            {2, 2}, {}, [&](const index_t &) { ++rd_init_calls; });
    auto rs1 = streams.add(
            {3, 2}, {}, [&](const index_t &) { ++rs1_init_calls; });
    auto rs2 = streams.add(
            {3, 2}, {}, [&](const index_t &) { ++rs2_init_calls; });

    // Stages kb=3, then 2, then 1, as a K=3 bound with max_ur=8 resolves
    mock_loop_t k({3, 2, 1});

    int dense_calls = 0;
    int inits_at_hook = -1, dense_at_hook = -1, hook_calls = 0;
    reduce(rd, rs1, rs2, k, stream::policy_t::zero(),
            [&](const index_t &, const index_t &) { ++dense_calls; }, [&] {
        ++hook_calls;
        inits_at_hook = rd_init_calls + rs1_init_calls + rs2_init_calls;
        dense_at_hook = dense_calls;
    });

    // The after-init hook runs once, after every init stream and before any
    // dense position
    EXPECT_EQ(hook_calls, 1);
    EXPECT_EQ(inits_at_hook, 2 * 2 + 3 * 2 + 3 * 2);
    EXPECT_EQ(dense_at_hook, 0);

    // Each stream's init closure fires once per position of its own shape
    // (rd: 2x2, rs1/rs2: 3x2) -- interleaved() drives every entry across
    // its own full shape, not once per operand
    EXPECT_EQ(rd_init_calls, 2 * 2);
    EXPECT_EQ(rs1_init_calls, 3 * 2);
    EXPECT_EQ(rs2_init_calls, 3 * 2);

    // K positions emitted: 3 + 2 + 1 across the stages (each resized to its
    // own width) x M=2 x N=2 -- if a stage's resize() failed to narrow the K
    // axis (e.g. leaving the original K=3 untouched), this would over-count
    EXPECT_EQ(dense_calls, (3 + 2 + 1) * 2 * 2);

    // rs1/rs2 come back at the shape they were handed in
    EXPECT_EQ(rs1.size(0), 3);
    EXPECT_EQ(rs2.size(0), 3);
}

} // namespace rvjit
