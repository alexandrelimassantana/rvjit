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
#include <string>

#include "gtest/gtest.h"

#include "env.hpp"
#include "loop_mock.hpp"
#include "rv64_jit_test_generator.hpp"
#include "stream_mock.hpp"

namespace rvjit {
namespace rvv {

using namespace Xbyak_riscv;

/// set_vtype drives the emitter from the housed `vtype`'s fields
TEST(rvv_config, set_vtype_uses_housed_vtype) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg avl = a0;
        const Reg out = a1;
        const Reg gvl = a2;

        model_env_t env(gen);
        env.pm.vtype.for_uniform(SEW::e8, LMUL::m1);
        env.pm.set_vtype(gvl, avl, vtype_t::cfg_t::input);
        gen.sd(gvl, out, 0);
        gen.ret();
    });

    const auto avl = 8;
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    kernel(avl, &out);
    EXPECT_EQ(out, avl);
}

/// The model's reduce defers each macc by the memory latency of its vpu
TEST(rvv_config, reduce_delays_by_memory_latency) {
    vpu_t vpu;
    vpu.from_params(128, 32 * 1024, 64, 2);
    host_model_env_t env(buffer::architecture_t::default_list(), vpu);

    std::string log;
    const auto record = [&](char c, int i) {
        log += c;
        log += std::to_string(i);
        log += ' ';
    };

    // B: {K, N} = {1, 4}; A: {K, M} = {1, 1}
    mock_streams_t streams;
    auto c = streams.add({1, 4});
    auto b = streams.add({1, 4}, [&](const index_t &i) { record('B', i[1]); });
    auto a = streams.add({1, 1}, [&](const index_t &i) { record('A', i[1]); });
    mock_loop_t k({1});

    const auto macc
            = [&](const index_t &, const index_t &bi) { record('c', bi[1]); };
    env.pm.streams().reduce()(c, b, a, k, stream::policy_t::memory(), macc);

    // Two productions run ahead of each macc; the last position flushes
    EXPECT_EQ(log, "B0 A0 B1 B2 c0 B3 c1 c2 c3 ");
}

} // namespace rvv
} // namespace rvjit
