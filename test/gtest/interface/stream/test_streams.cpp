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

#include "rvjit/interface/stream/stream_impl.hpp"

namespace rvjit {

namespace {

// A single 2x3 stream recording the positions it is driven at
struct recording_streams_t : public stream::stream_impl_t {
    int ndims() const override { return 2; }
    int size(int d) const override { return d == 0 ? 2 : 3; }

    std::vector<int> produced, inited;

protected:
    void produce_(const index_t &idx) override {
        produced.push_back(idx[0] * 10 + idx[1]);
    }
    void init_(const index_t &idx) override {
        inited.push_back(idx[0] * 10 + idx[1]);
    }
};

} // namespace

TEST(streams, emit_sweeps_in_ascending_nested_order) {
    recording_streams_t s;
    s.emit();
    EXPECT_EQ(s.produced, (std::vector<int> {0, 1, 2, 10, 11, 12}));
    EXPECT_TRUE(s.inited.empty());
}

TEST(streams, emit_init_sweeps_in_ascending_nested_order) {
    recording_streams_t s;
    s.emit_init();
    EXPECT_EQ(s.inited, (std::vector<int> {0, 1, 2, 10, 11, 12}));
    EXPECT_TRUE(s.produced.empty());
}

TEST(streams, iteration_skips_positions_outside_shape) {
    recording_streams_t s;
    s.iteration({2, 0});
    s.iteration({0, 3});
    s.iteration({-1, 0});
    s.iteration({1, 2});
    EXPECT_EQ(s.produced, (std::vector<int> {12}));
}

} // namespace rvjit
