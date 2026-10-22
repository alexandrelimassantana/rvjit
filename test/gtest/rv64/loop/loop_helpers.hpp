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

#ifndef RVJIT_TEST_RV64_LOOP_LOOP_HELPERS_HPP
#define RVJIT_TEST_RV64_LOOP_LOOP_HELPERS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// Loop sizes exercised by every correctness suite in this file
///
/// @details The suites count:
///     - trips: the times `cb` fires (`passes`)
///     - iterations: the sum of work `cb` processes (`total`)
inline ::testing::internal::ParamGenerator<int64_t> loop_sizes() {
    static const int64_t sizes[] = {0, 1, 2, 3, 4, 15, 16, 17};
    return ::testing::ValuesIn(sizes);
}

/// The `unroll` factor list `new_loop()` derives for a `runtime`/`less_eq`
/// parameter bounded by `max_ur`: `max_ur` itself, then its power-of-two
/// breakdown
inline std::vector<int> pow2_factors(int max_ur) {
    int buf[8];
    int n = 0;
    if (max_ur > 0 && !is_pow2(max_ur)) buf[n++] = max_ur;
    n += pow2_breakdown(max_ur, buf + n, 8 - n);
    return std::vector<int>(buf, buf + n);
}

/// A register standing in for a runtime size the test never executes
inline Reg dummy_reg() {
    return a0;
}

/// Builds a loop in `mode`, optionally with `u` as its unrolling, drawing
/// from `on` or a shared environment
inline rv64::loop::loop_t make_desc(rv64::loop::loop_t::mode_t mode,
        loop::unroll_t u = loop::unroll_t(), host_env_t *on = nullptr) {
    static host_env_t shared;
    host_env_t &env = on ? *on : shared;
    rv64::loop::loop_t d(
            env.e, env.arith, env.cf, env.pool, env.stack, env.xfer);
    d.mode = mode;
    if (u.is_valid()) d.factors = u;
    return d;
}

inline rv64::loop::loop_t literal_desc() {
    return make_desc(rv64::loop::loop_t::mode_t::literal);
}

inline rv64::loop::loop_t switch_desc(loop::unroll_t u = loop::unroll_t()) {
    return make_desc(rv64::loop::loop_t::mode_t::switch_, u);
}

inline rv64::loop::loop_t unrolled_desc(loop::unroll_t u = loop::unroll_t(1)) {
    return make_desc(rv64::loop::loop_t::mode_t::unroll, u);
}

inline rv64::loop::loop_t unroll_switch_desc(
        loop::unroll_t u = loop::unroll_t()) {
    return make_desc(rv64::loop::loop_t::mode_t::unroll_switch, u);
}

namespace {

/// Trips and iterations of a simulated loop
///
/// @note A trip is one `cb` call; an iteration is one unit of work
struct unroll_result_t {
    int64_t total = 0;
    int64_t passes = 0;
};

/// Reference re-implementation of unroll()'s trip/iteration counting,
/// independent of the component under test
inline unroll_result_t simulate_unroll(
        const std::vector<int> &factors, int64_t n, int granularity) {
    unroll_result_t r;
    int64_t remaining = n;
    for (int ur : factors) {
        while (remaining >= static_cast<int64_t>(ur) * granularity) {
            r.passes += 1;
            for (int i = 0; i < ur; ++i) {
                const int64_t granted
                        = std::min<int64_t>(remaining, granularity);
                r.total += granted;
                remaining -= granted;
            }
        }
        if (ur == 1) {
            while (remaining > 0) {
                r.passes += 1;
                const int64_t granted
                        = std::min<int64_t>(remaining, granularity);
                r.total += granted;
                remaining -= granted;
            }
        }
    }
    return r;
}

/// Runs `k`, comparing the resulting passes/total against `expect`
///
/// @note The trips-and-iterations correctness check shared by every
///     unroll()/unroll_switch() correctness suite below
template <typename... Args>
inline void expect_trips_and_iterations(
        rv64_jit_probe_t &k, const unroll_result_t &expect, Args &&...args) {
    ASSERT_TRUE(k.create_kernel());
    int64_t total = -1, passes = -1;
    run_timed(k, std::forward<Args>(args)..., &total, &passes);
    EXPECT_EQ(passes, expect.passes);
    EXPECT_EQ(total, expect.total);
}

} // namespace

/// Reference re-implementation of unroll_switch()'s trip/iteration counting
///
/// @note The tail drains only whole `granularity` chunks, leaving any smaller
///     remainder undrained
inline unroll_result_t simulate_unroll_and_switch(
        const std::vector<int> &factors, int64_t n, int granularity) {
    unroll_result_t r = simulate_unroll(factors, n, granularity);
    const int64_t leftover = n - r.total;
    const int64_t chunks = leftover / granularity;
    if (chunks > 0) {
        r.total += chunks * granularity;
        r.passes += 1;
    }
    return r;
}

} // namespace rvjit

#endif
