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

#ifndef RVJIT_TEST_BENCH_COMMON_TIMING_HPP
#define RVJIT_TEST_BENCH_COMMON_TIMING_HPP

#include <chrono>
#include <functional>

namespace rvjit_bench {

struct timing_result_t {
    double best_s = -1.0;
    double avg_s = 0.0;
    int reps = 0;
};

/// Soft benchmark: `warmup` untimed calls, then `reps` timed calls of `fn`
///
/// @note Reports the best and average wall-clock time
/// @note This is not a statistically rigorous engine but enough to see changes
inline timing_result_t time_it(
        const std::function<void()> &fn, int reps, int warmup = 3) {
    for (int i = 0; i < warmup; ++i)
        fn();

    timing_result_t r;
    r.reps = reps;
    double total = 0.0;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        total += s;
        if (r.best_s < 0.0 || s < r.best_s) r.best_s = s;
    }
    r.avg_s = total / reps;
    return r;
}

inline double gflops(double ops, double seconds) {
    return seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
}

} // namespace rvjit_bench

#endif
