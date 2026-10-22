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

#ifndef RVJIT_TEST_BENCH_COMMON_COMPARE_HPP
#define RVJIT_TEST_BENCH_COMMON_COMPARE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

namespace rvjit_bench {

struct compare_result_t {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    int64_t n_bad = 0;
    int64_t n_total = 0;

    bool passed() const { return n_bad == 0; }
};

using on_bad_t = std::function<void(int64_t idx, double got, double ref)>;

/// Compares `got[i]` against `ref[i]` for `i` in `[0, n)` under `atol`/`rtol`
///
/// @details `on_bad`, if set, is invoked once per out-of-tolerance element
inline compare_result_t compare(const double *got, const double *ref, int64_t n,
        double atol, double rtol, const on_bad_t &on_bad = nullptr) {
    compare_result_t r;
    r.n_total = n;
    for (int64_t i = 0; i < n; ++i) {
        const double err = std::fabs(got[i] - ref[i]);
        const double rel = err / std::max(std::fabs(ref[i]), 1e-6);
        r.max_abs_err = std::max(r.max_abs_err, err);
        r.max_rel_err = std::max(r.max_rel_err, rel);
        if (err > atol + rtol * std::fabs(ref[i])) {
            ++r.n_bad;
            if (on_bad) on_bad(i, got[i], ref[i]);
        }
    }
    return r;
}

} // namespace rvjit_bench

#endif
