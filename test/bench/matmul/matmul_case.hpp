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

#ifndef RVJIT_TEST_BENCH_MATMUL_CASE_HPP
#define RVJIT_TEST_BENCH_MATMUL_CASE_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include "bench/common/dtype.hpp"
#include "rvjit/interface/isa/parameter.hpp"

namespace rvjit_bench {

/// How a size/stride parameter is handed to the kernel
enum class source_kind_t { literal, blocked, less_eq, runtime };

inline const char *to_string(source_kind_t k) {
    switch (k) {
        case source_kind_t::literal: return "literal";
        case source_kind_t::blocked: return "blocked";
        case source_kind_t::less_eq: return "less_eq";
        case source_kind_t::runtime: return "runtime";
    }
    return "?";
}

/// Where a non-literal size lives when the kernel starts
enum class size_at_t {
    reg, // A register the kernel loads from the args structure
    mem, // The args structure itself, which the backend reads
};

inline const char *to_string(size_at_t s) {
    return s == size_at_t::mem ? "mem" : "reg";
}

/// Epilogue features a case compiles in, matching
/// rvjit::linalg::matmul::codegen_flags_t bit for bit
enum post_flags_t : int {
    post_alpha = 1,
    post_beta = 2,
    post_bias = 4,
    post_postop = 8, // a native test post-op scaling the result by 4
};

/// How the runtime flag word reaches the kernel
enum class rt_kind_t { none, literal, runtime };

inline const char *to_string(rt_kind_t k) {
    switch (k) {
        case rt_kind_t::none: return "none";
        case rt_kind_t::literal: return "literal";
        case rt_kind_t::runtime: return "runtime";
    }
    return "?";
}

/// "alpha+beta"-style name of a set of post flags
inline std::string post_name(int post) {
    if (post == 0) return "none";
    std::string s;
    const auto add = [&](int bit, const char *n) {
        if (!(post & bit)) return;
        if (!s.empty()) s += '+';
        s += n;
    };
    add(post_alpha, "alpha");
    add(post_beta, "beta");
    add(post_bias, "bias");
    add(post_postop, "postop");
    return s;
}

/// One fully-resolved matmul problem, plus the rvv::linalg::matmul::matmul_t
/// tuning knobs and hardware-model parameters it was generated under
struct matmul_case_t {
    std::string name;

    int64_t m = 0, n = 0, k = 0;
    bool trans_a = false, trans_b = false;

    // Data types of A, B and C; the products accumulate in `dtacc`, wider
    // than A for a widening matmul and than C for a narrowed output
    dtype_t dta = dtype_t::f32, dtb = dtype_t::f32, dtc = dtype_t::f32;
    dtype_t dtacc = dtype_t::f32;

    // The type B has in its register once loaded, wider than `dtb` when the
    // load extends it; none is `dtb`
    dtype_t dtbreg = dtype_t::none;

    // The type A has in its registers, converted from `dta` as it is loaded;
    // none is `dta`
    dtype_t dtareg = dtype_t::none;

    // The type the epilogue works in, converted from `dtacc` first; none is
    // `dtacc`
    dtype_t dtepi = dtype_t::none;

    // The bias array's type, converted to the epilogue's as it is loaded;
    // none is the epilogue's. A scalar bias is a single element for all M
    dtype_t dtbias = dtype_t::none;
    bool bias_scalar = false;

    dtype_t areg() const { return dtareg == dtype_t::none ? dta : dtareg; }
    dtype_t epi() const { return dtepi == dtype_t::none ? dtacc : dtepi; }
    dtype_t bias_dt() const { return dtbias == dtype_t::none ? epi() : dtbias; }

    // M and N share one source kind, K has its own (literal by default,
    // being the reduction dimension). `blocked` values are the block size
    // the plan pins: N and K as is, M as whole vectors of `vl` elements;
    // `less_eq` values are the chunk bound the host driver tiles M/N by, and
    // the K one call may carry
    source_kind_t shape_mn = source_kind_t::literal;
    source_kind_t shape_k = source_kind_t::literal;
    int m_blk = 0, n_blk = 0, k_blk = 0;
    int m_max = 0, n_max = 0, k_max = 0;
    size_at_t size_at = size_at_t::reg;

    // Elements one vector register group holds: the unit M blocks pin in
    int vl = 1;

    // Leading dimensions share one source kind (literal or runtime); an
    // explicit ld (elements) of 0 means the packed one
    source_kind_t ld_src = source_kind_t::literal;
    int64_t lda = 0, ldb = 0, ldc = 0;

    // Register-pressure stress: caps the int registers configure() sees as
    // available (0 = unconstrained/default)
    int max_int_regs = 0;

    // Vector register grouping override: 1, 2, 4 or 8 (0 = let it decide)
    int lmul = 0;

    // vpu_t::from_params hardware-model parameters; vlen itself always
    // comes from the running hardware/qemu's own vlenb CSR, not from here
    int cache_size = 32768;
    int cache_line_size = 64;
    int cache_latency = 4;

    int reps = 20;

    // Epilogue: C = alpha * A * B + beta * C + bias, then a post-op chain.
    // `post` holds the features compiled in; the runtime flag word reaches the
    // kernel as `rt` says and enables `rt_mask & post` (-1: all of them).
    // `scalar_src` says whether alpha/beta are literals or loaded from the
    // args structure. `kblock` splits K across calls (0: one call), the first
    // call applying the user's beta, the later ones accumulating, and the last
    // one adding the bias and running the post-op
    int post = 0;
    double alpha = 2.0, beta = 3.0;
    source_kind_t scalar_src = source_kind_t::literal;
    rt_kind_t rt = rt_kind_t::none;
    int rt_mask = -1;
    int64_t kblock = 0;

    /// The post flags the runtime flag word leaves enabled
    int enabled_post() const {
        return rt == rt_kind_t::none ? post : post & rt_mask;
    }

    bool is_widening() const { return areg() != dtacc; }

    bool narrows() const { return epi() != dtc; }

    /// Leading dimensions in elements, defaulting to the packed layout
    int64_t lda_elems() const { return lda > 0 ? lda : (trans_a ? k : m); }
    int64_t ldb_elems() const { return ldb > 0 ? ldb : (trans_b ? n : k); }
    int64_t ldc_elems() const { return ldc > 0 ? ldc : m; }

    /// Whether the explicit leading dimensions can hold the shape
    bool lds_fit() const {
        return lda_elems() >= (trans_a ? k : m)
                && ldb_elems() >= (trans_b ? n : k) && ldc_elems() >= m;
    }

    /// A single-line, directly reproducible CLI invocation for this case
    std::string repro() const {
        char buf[768];
        std::snprintf(buf, sizeof(buf),
                "--dt=%s:%s:%s --acc=%s --breg=%s --transA=%d --transB=%d "
                "--shape=%s "
                "--kshape=%s "
                "--ld=%s --size_at=%s "
                "--m_blk=%d --n_blk=%d --k_blk=%d --m_max=%d --n_max=%d "
                "--k_max=%d "
                "--lda=%lld --ldb=%lld --ldc=%lld --max_int_regs=%d "
                "--lmul=%d %lldx%lldx%lld",
                dtype_name(dta), dtype_name(dtb), dtype_name(dtc),
                dtype_name(dtacc),
                dtbreg == dtype_t::none ? "none" : dtype_name(dtbreg),
                trans_a ? 1 : 0, trans_b ? 1 : 0, to_string(shape_mn),
                to_string(shape_k), to_string(ld_src), to_string(size_at),
                m_blk, n_blk, k_blk, m_max, n_max, k_max,
                static_cast<long long>(lda), static_cast<long long>(ldb),
                static_cast<long long>(ldc), max_int_regs, lmul,
                static_cast<long long>(m), static_cast<long long>(n),
                static_cast<long long>(k));
        std::string s(buf);
        if (post != 0 || kblock != 0) {
            char ebuf[256];
            std::snprintf(ebuf, sizeof(ebuf),
                    " --post=%s --alpha=%g --beta=%g --scalar=%s --rt=%s "
                    "--rtmask=%d --kblock=%lld",
                    post_name(post).c_str(), alpha, beta, to_string(scalar_src),
                    to_string(rt), rt_mask, static_cast<long long>(kblock));
            s.insert(s.rfind(' ' + std::to_string(m) + 'x'), ebuf);
        }
        if (dtareg != dtype_t::none || dtepi != dtype_t::none
                || dtbias != dtype_t::none || bias_scalar) {
            char tbuf[160];
            std::snprintf(tbuf, sizeof(tbuf),
                    " --areg=%s --epi=%s --bias_dt=%s --bias_kind=%s",
                    dtareg == dtype_t::none ? "none" : dtype_name(dtareg),
                    dtepi == dtype_t::none ? "none" : dtype_name(dtepi),
                    dtbias == dtype_t::none ? "none" : dtype_name(dtbias),
                    bias_scalar ? "scalar" : "per_m");
            s.insert(s.rfind(' ' + std::to_string(m) + 'x'), tbuf);
        }
        if (!name.empty()) s += "_n\"" + name + "\"";
        return s;
    }
};

} // namespace rvjit_bench

#endif
