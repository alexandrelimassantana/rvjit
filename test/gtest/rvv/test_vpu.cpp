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

#include "rvjit/implementation/rvv/rvv.hpp"

namespace rvjit {
namespace rvv {

/// Expected vector latency is vlenb/vlaneb when the vector lane width is set
TEST(vpu, vector_latency_divides) {
    const vpu_t vpu(64, 16, LMUL::m8, LMUL::m1, 0, 0, 0);
    EXPECT_EQ(vpu.vector_latency(), 4);
}

/// Expected vector latency is zero when the vector length is not set
TEST(vpu, vector_latency_without_vector_lanes) {
    const vpu_t vpu(64, 0, LMUL::m8, LMUL::m1, 0, 0, 0);
    EXPECT_EQ(vpu.vector_latency(), 1);
}

/// Expected memory latency forwards informed values
TEST(vpu, memory_latency) {
    const vpu_t vpu(64, 16, LMUL::m8, LMUL::m1, 32768, 64, 5);
    EXPECT_EQ(vpu.vector_latency(), 4);
    EXPECT_EQ(vpu.memory_latency(), 5);
}

/// Smoke test for initialization with common parameter values
TEST(vpu, from_params_smoke) {
    const auto vlen = 256;
    const auto cache = 32768;
    const auto line = 64;
    const auto latency = 5;

    vpu_t model;
    model.from_params(vlen, cache, line, latency);

    EXPECT_GT(model.vlenb, 0);
    EXPECT_GT(model.vlaneb, 0);
    EXPECT_GT(model.cache_line_size, 0);
    EXPECT_GT(model.cache_size, 0);
}

} // namespace rvv
} // namespace rvjit
