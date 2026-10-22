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

#ifndef RVJIT_TEST_BENCH_MATMUL_REF_HPP
#define RVJIT_TEST_BENCH_MATMUL_REF_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "bench/common/dtype.hpp"
#include "matmul_case.hpp"

namespace rvjit_bench {

/// Byte the buffers' padding is filled with, so a stray read or write there
/// is visible
constexpr uint8_t pad_byte = 0x5b;

/// Bytes of padding past the end of C, catching a store beyond the last column
constexpr size_t guard_bytes = 512;

using bytes_t = std::vector<uint8_t>;

/// Row/column layout accessors matching matmul_kernel_t's own convention,
/// each returning the element index of a position
struct matmul_layout_t {
    const matmul_case_t &c;

    int64_t lda() const { return c.lda_elems(); }
    int64_t ldb() const { return c.ldb_elems(); }
    int64_t ldc() const { return c.ldc_elems(); }

    /// Elements each buffer spans, padding included
    size_t a_size() const { return (c.trans_a ? c.m : c.k) * lda(); }
    size_t b_size() const { return (c.trans_b ? c.k : c.n) * ldb(); }
    size_t c_size() const { return c.n * ldc(); }

    int64_t a_at(int64_t m, int64_t k) const {
        return c.trans_a ? m * lda() + k : k * lda() + m;
    }
    int64_t b_at(int64_t k, int64_t n) const {
        return c.trans_b ? k * ldb() + n : n * ldb() + k;
    }
    int64_t c_at(int64_t m, int64_t n) const { return n * ldc() + m; }
};

/// Largest magnitude an input takes: floats stay within 1, and an integer
/// input keeps the typical accumulation within the accumulator's range
///
/// @note An integer B extended by its load takes its whole range, so the
///     extension is visible
inline int64_t input_bound(
        const matmul_case_t &c, dtype_t dt, bool operand_b = false) {
    if (dtype_is_float(dt)) return 1;
    const double room = static_cast<double>(dtype_int_max(c.dtacc)) * 0.5
            / static_cast<double>(std::max<int64_t>(c.k, 1));
    const int64_t r = static_cast<int64_t>(std::sqrt(room));
    const bool extended = operand_b && c.dtbreg != dtype_t::none;
    const int64_t cap = extended ? dtype_int_max(dt) : 100;
    return std::max<int64_t>(1, std::min<int64_t>(r, cap));
}

/// `v` converted to the output type `out`: floats round to nearest even and
/// integers saturate, a float rounding to the nearest even integer first
inline double narrow_value(double v, dtype_t out) {
    if (dtype_is_float(out)) {
        uint8_t tmp[8];
        store_elem(tmp, 0, out, v);
        return load_elem(tmp, 0, out);
    }
    const double hi = static_cast<double>(dtype_int_max(out));
    const double lo = dtype_is_signed(out) ? -hi - 1.0 : 0.0;
    return std::min(hi, std::max(lo, std::nearbyint(v)));
}

/// The bit pattern a kernel loads for the scalar `v` of type `dt`
inline uint64_t scalar_bits(double v, dtype_t dt) {
    switch (dt) {
        case dtype_t::f16: return f32_to_f16_bits(static_cast<float>(v));
        case dtype_t::f32: {
            const float f = static_cast<float>(v);
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        }
        case dtype_t::f64: {
            uint64_t u;
            std::memcpy(&u, &v, sizeof(u));
            return u;
        }
        default: return static_cast<uint64_t>(static_cast<int64_t>(v));
    }
}

/// Fills A/B (padding with `pad_byte`) with a fixed-seed uniform
/// distribution, and `Cref` with the double-precision scalar reference
/// of `alpha * A * B + beta * C + bias`, then the post-op, over the
/// dtype-rounded inputs; `C` receives the initial output and `bias` the bias
/// vector
///
/// @note Integer products accumulate in the accumulator's own wrapping range
/// @note A zero beta leaves C unread, which the NaN-filled C of a float case
///     checks; a literal zero alpha does likewise to an infinite A
inline void fill_and_reference(const matmul_layout_t &layout, bytes_t &a_data,
        bytes_t &b_data, bytes_t &c_data, bytes_t &bias,
        std::vector<double> &c_ref) {
    const auto &c = layout.c;
    a_data.assign(layout.a_size() * dtype_bytes(c.dta), pad_byte);
    b_data.assign(layout.b_size() * dtype_bytes(c.dtb), pad_byte);
    std::mt19937 rng(42);
    const auto draw = [&](dtype_t dt, bool operand_b = false) -> double {
        if (dtype_is_float(dt)) {
            std::uniform_real_distribution<double> d(-1.0, 1.0);
            return d(rng);
        }
        const int64_t r = input_bound(c, dt, operand_b);
        std::uniform_int_distribution<int64_t> d(
                dtype_is_signed(dt) ? -r : 0, r);
        return static_cast<double>(d(rng));
    };
    for (int64_t k = 0; k < c.k; ++k)
        for (int64_t m = 0; m < c.m; ++m)
            store_elem(a_data.data(), layout.a_at(m, k), c.dta, draw(c.dta));
    for (int64_t n = 0; n < c.n; ++n)
        for (int64_t k = 0; k < c.k; ++k)
            store_elem(
                    b_data.data(), layout.b_at(k, n), c.dtb, draw(c.dtb, true));

    // An integer epilogue wraps in the accumulator's range; an integer
    // accumulation scaled in a float epilogue converts exactly first
    const bool ints = !dtype_is_float(c.epi());
    const bool int_acc = !dtype_is_float(c.dtacc);
    const int post = c.kblock > 0 ? c.post : c.enabled_post();
    const bool has_alpha = post & post_alpha;
    const bool has_beta = (post & post_beta) && c.beta != 0.0;
    const bool has_bias = post & post_bias;

    // A literal zero alpha must keep an infinite input out of C
    if (has_alpha && c.alpha == 0.0 && c.scalar_src == source_kind_t::literal
            && dtype_is_float(c.dta) && c.m > 0 && c.k > 0)
        store_elem(a_data.data(), layout.a_at(0, 0), c.dta, INFINITY);

    // The initial C and the bias, over the C block (the rest stays padding)
    std::vector<double> cin(static_cast<size_t>(c.m * c.n), 0.0);
    std::vector<double> bvec(static_cast<size_t>(c.m), 0.0);
    const dtype_t bdt = c.bias_dt();
    if (post & post_beta) {
        for (int64_t n = 0; n < c.n; ++n)
            for (int64_t m = 0; m < c.m; ++m) {
                // A zero beta never reads C, so poison it
                const double v = c.beta == 0.0 && !ints
                        ? std::numeric_limits<double>::quiet_NaN()
                        : draw(c.dtc);
                store_elem(c_data.data(), layout.c_at(m, n), c.dtc, v);
                // The kernel reads C back as stored, rounded to its type
                cin[n * c.m + m]
                        = load_elem(c_data.data(), layout.c_at(m, n), c.dtc);
            }
    }
    if (has_bias) {
        bias.assign((c.m + 8) * dtype_bytes(bdt), pad_byte);
        const int64_t elems = c.bias_scalar ? 1 : c.m;
        for (int64_t m = 0; m < elems; ++m)
            store_elem(bias.data(), m, bdt, draw(bdt));
        for (int64_t m = 0; m < c.m; ++m)
            bvec[m] = load_elem(bias.data(), c.bias_scalar ? 0 : m, bdt);
    }

    const auto epilogue = [&](double v, int64_t m, int64_t n) {
        if (ints) {
            int64_t i = static_cast<int64_t>(v);
            if (has_alpha)
                i = wrap_int(i * static_cast<int64_t>(c.alpha), c.dtacc);
            if (has_beta)
                i = wrap_int(i
                                + static_cast<int64_t>(c.beta)
                                        * static_cast<int64_t>(
                                                cin[n * c.m + m]),
                        c.dtacc);
            if (has_bias)
                i = wrap_int(i + static_cast<int64_t>(bvec[m]), c.dtacc);
            if (post & post_postop) i = wrap_int(i * 4, c.dtacc);
            return static_cast<double>(i);
        }
        if (has_alpha)
            v = c.alpha == 0.0 && c.scalar_src == source_kind_t::literal
                    ? 0.0
                    : v * c.alpha;
        if (has_beta) v += c.beta * cin[n * c.m + m];
        if (has_bias) v += bvec[m];
        if (post & post_postop) v *= 4.0;
        return v;
    };

    for (int64_t m = 0; m < c.m; ++m) {
        for (int64_t n = 0; n < c.n; ++n) {
            double acc = 0.0;
            int64_t iacc = 0;
            for (int64_t k = 0; k < c.k; ++k) {
                const double a
                        = load_elem(a_data.data(), layout.a_at(m, k), c.dta);
                const double b
                        = load_elem(b_data.data(), layout.b_at(k, n), c.dtb);
                if (int_acc)
                    iacc = wrap_int(
                            iacc + static_cast<int64_t>(a * b), c.dtacc);
                else
                    acc += a * b;
            }
            c_ref[n * c.m + m] = narrow_value(
                    epilogue(int_acc ? static_cast<double>(iacc) : acc, m, n),
                    c.dtc);
        }
    }
}

/// Absolute and relative tolerance of the C comparison
///
/// @note Integer results are exact; a float accumulator rounds once per K step
struct tolerance_t {
    double atol, rtol;
};

inline tolerance_t tolerance_of(const matmul_case_t &c) {
    // A float epilogue rounds to an integer output once, maybe at a tie
    if (!dtype_is_float(c.dtc))
        return {dtype_is_float(c.epi()) ? 1.0 : 0.0, 0.0};
    if (c.dtc == dtype_t::f32) return {1e-4, 1e-2};
    if (c.narrows()) {
        // The products accumulate wide and round once, to the output type
        return {1e-3, 4 * dtype_eps(c.dtc)};
    }
    if (c.post != 0) {
        // Each epilogue step rounds once more, and scales the magnitude
        const double eps = dtype_eps(c.dtc);
        return {std::max(1e-6, 64.0 * static_cast<double>(c.k + 4) * eps),
                64 * eps};
    }
    const double eps = dtype_eps(c.dtc);
    return {std::max(1e-9, 2.0 * static_cast<double>(c.k) * eps), 20 * eps};
}

/// Number of padding bytes of C (outside the `M x N` block, and the guard past
/// its end) that no longer hold the pad byte
inline int64_t count_padding_writes(
        const matmul_layout_t &layout, const bytes_t &c_data) {
    const auto &c = layout.c;
    const int w = dtype_bytes(c.dtc);
    int64_t bad = 0;
    for (size_t i = layout.c_size() * w; i < c_data.size(); ++i)
        bad += c_data[i] != pad_byte;
    for (int64_t n = 0; n < c.n; ++n)
        for (int64_t m = c.m; m < layout.ldc(); ++m)
            for (int b = 0; b < w; ++b)
                bad += c_data[(n * layout.ldc() + m) * w + b] != pad_byte;
    return bad;
}

} // namespace rvjit_bench

#endif
