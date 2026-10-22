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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/common/asm_dump.hpp"
#include "bench/common/cli.hpp"
#include "bench/common/compare.hpp"
#include "bench/common/csr.hpp"
#include "bench/common/timing.hpp"

#include "dtype_probe.hpp"
#include "matmul_case.hpp"
#include "matmul_kernel.hpp"
#include "ref_matmul.hpp"

namespace rvjit_bench {

enum class run_mode_t { correctness, perf };

/// What a case expects of the backend's plan: `pass` (must JIT and be
/// correct), `any` (a clean plan rejection is tolerated) or `reject`
enum class expect_t { pass, any, reject };

enum class result_t { pass, fail, rejected, skipped };

struct dtypes_t {
    dtype_t a, b, c;
};

/// Sticky CLI state (benchdnn's own `settings_t`): every field is a list, so
/// a positional dims spec cross-products against whatever's currently set
struct matmul_settings_t {
    std::vector<dtypes_t> dt {{dtype_t::f32, dtype_t::f32, dtype_t::f32}};
    // The accumulation type; none is C's own
    std::vector<dtype_t> acc {dtype_t::none};
    // The type B has in its register; none is B's own
    std::vector<dtype_t> breg {dtype_t::none};
    // The type A has in its registers; none is A's own
    std::vector<dtype_t> areg {dtype_t::none};
    // The epilogue's type; none is the accumulation type
    std::vector<dtype_t> epi {dtype_t::none};
    // The bias array's type; none is the epilogue's
    std::vector<dtype_t> bias_dt {dtype_t::none};
    // Whether the bias is a single element
    std::vector<bool> bias_scalar {false};
    std::vector<bool> trans_a {false};
    std::vector<bool> trans_b {false};
    std::vector<source_kind_t> shape {source_kind_t::literal};
    std::vector<source_kind_t> kshape {source_kind_t::literal};
    std::vector<source_kind_t> ld {source_kind_t::literal};
    std::vector<size_at_t> size_at {size_at_t::reg};
    // Sizes are `eval_vl_expr` expressions, resolved per case
    std::vector<std::string> m_blk {"0"}, n_blk {"0"}, k_blk {"0"};
    std::vector<std::string> m_max {"0"}, n_max {"0"}, k_max {"0"};
    std::vector<std::string> lda {"0"}, ldb {"0"}, ldc {"0"};
    std::vector<int64_t> max_int_regs {0};
    std::vector<int64_t> lmul {0};
    std::vector<int> post {0};
    std::vector<double> alpha {2.0}, beta {3.0};
    std::vector<source_kind_t> scalar {source_kind_t::literal};
    std::vector<rt_kind_t> rt {rt_kind_t::none};
    std::vector<int64_t> rtmask {-1};
    std::vector<int64_t> kblock {0};
    expect_t expect = expect_t::pass;
    int64_t reps = 20;
    int64_t cache_size = 32768;
    int64_t cache_line_size = 64;
    int64_t cache_latency = 4;
};

bool parse_source_list(const std::string &value,
        std::vector<source_kind_t> &out, bool allow_blocked_less_eq) {
    out.clear();
    for (const auto &v : split_csv(value)) {
        if (v == "literal") {
            out.push_back(source_kind_t::literal);
        } else if (v == "runtime") {
            out.push_back(source_kind_t::runtime);
        } else if (allow_blocked_less_eq && v == "blocked") {
            out.push_back(source_kind_t::blocked);
        } else if (allow_blocked_less_eq && v == "less_eq") {
            out.push_back(source_kind_t::less_eq);
        } else {
            std::fprintf(
                    stderr, "error: invalid source kind '%s'\n", v.c_str());
            return false;
        }
    }
    return !out.empty();
}

/// Parses `none` or `+`-joined `alpha`/`beta`/`bias`/`postop` feature sets
bool parse_post_list(const std::string &value, std::vector<int> &out) {
    out.clear();
    for (const auto &v : split_csv(value)) {
        int post = 0;
        std::string cur;
        for (size_t i = 0; i <= v.size(); ++i) {
            if (i < v.size() && v[i] != '+') {
                cur += v[i];
                continue;
            }
            if (cur == "alpha")
                post |= post_alpha;
            else if (cur == "beta")
                post |= post_beta;
            else if (cur == "bias")
                post |= post_bias;
            else if (cur == "postop")
                post |= post_postop;
            else if (cur != "none") {
                std::fprintf(
                        stderr, "error: invalid --post '%s'\n", cur.c_str());
                return false;
            }
            cur.clear();
        }
        out.push_back(post);
    }
    return !out.empty();
}

bool parse_size_at_list(const std::string &value, std::vector<size_at_t> &out) {
    out.clear();
    for (const auto &v : split_csv(value)) {
        if (v == "reg")
            out.push_back(size_at_t::reg);
        else if (v == "mem")
            out.push_back(size_at_t::mem);
        else {
            std::fprintf(stderr, "error: invalid --size_at '%s'\n", v.c_str());
            return false;
        }
    }
    return !out.empty();
}

bool parse_rt_list(const std::string &value, std::vector<rt_kind_t> &out) {
    out.clear();
    for (const auto &v : split_csv(value)) {
        if (v == "none")
            out.push_back(rt_kind_t::none);
        else if (v == "literal")
            out.push_back(rt_kind_t::literal);
        else if (v == "runtime")
            out.push_back(rt_kind_t::runtime);
        else {
            std::fprintf(stderr, "error: invalid --rt '%s'\n", v.c_str());
            return false;
        }
    }
    return !out.empty();
}

std::vector<double> parse_double_list(const std::string &value) {
    std::vector<double> out;
    for (const auto &v : split_csv(value))
        out.push_back(std::atof(v.c_str()));
    return out;
}

/// Parses `a`, `a:c` or `a:b:c` dtype triples (`a` alone is every operand)
bool parse_dtypes_list(const std::string &value, std::vector<dtypes_t> &out) {
    out.clear();
    for (const auto &v : split_csv(value)) {
        std::vector<dtype_t> dts;
        std::string cur;
        for (size_t i = 0; i <= v.size(); ++i) {
            if (i == v.size() || v[i] == ':') {
                dtype_t dt;
                if (!parse_dtype(cur, dt)) {
                    std::fprintf(
                            stderr, "error: invalid dtype '%s'\n", cur.c_str());
                    return false;
                }
                dts.push_back(dt);
                cur.clear();
            } else {
                cur += v[i];
            }
        }
        if (dts.size() == 1)
            out.push_back({dts[0], dts[0], dts[0]});
        else if (dts.size() == 2)
            out.push_back({dts[0], dts[0], dts[1]});
        else if (dts.size() == 3)
            out.push_back({dts[0], dts[1], dts[2]});
        else
            return false;
    }
    return !out.empty();
}

/// Whether the case's source kinds have the values they need
bool case_is_consistent(const matmul_case_t &c) {
    const auto need = [&](source_kind_t k, int blk, int max, const char *dim) {
        if (k == source_kind_t::blocked && blk <= 0) {
            std::fprintf(stderr,
                    "error: --shape/--kshape=blocked needs "
                    "--%c_blk > 0\n",
                    dim[0]);
            return false;
        }
        if (k == source_kind_t::less_eq && max <= 0) {
            std::fprintf(stderr,
                    "error: --shape/--kshape=less_eq needs --%c_max > 0\n",
                    dim[0]);
            return false;
        }
        return true;
    };
    if (c.shape_k == source_kind_t::less_eq && c.k_max > 0
            && (c.kblock > 0 ? c.kblock : c.k) > c.k_max) {
        std::fprintf(stderr,
                "error: --kshape=less_eq needs each call's K (--kblock, else "
                "the whole K) within --k_max\n");
        return false;
    }
    if (c.ld_src == source_kind_t::blocked
            || c.ld_src == source_kind_t::less_eq) {
        std::fprintf(stderr, "error: --ld takes literal or runtime\n");
        return false;
    }
    if (c.narrows() && c.kblock > 0) {
        std::fprintf(stderr,
                "error: --kblock rounds C once per call, which the narrowed "
                "reference does not model\n");
        return false;
    }
    if (c.kblock > 0) {
        if (!(c.post & post_beta)
                || (c.shape_k != source_kind_t::runtime
                        && c.shape_k != source_kind_t::less_eq)
                || c.scalar_src != source_kind_t::runtime
                || c.rt != rt_kind_t::runtime || c.rt_mask != -1) {
            std::fprintf(stderr,
                    "error: --kblock needs --post with beta, "
                    "--kshape=runtime|less_eq, "
                    "--scalar=runtime, --rt=runtime and no --rtmask\n");
            return false;
        }
    }
    if (c.rt == rt_kind_t::none && c.rt_mask != -1) {
        std::fprintf(stderr, "error: --rtmask needs --rt=literal|runtime\n");
        return false;
    }
    if (!c.lds_fit()) {
        std::fprintf(stderr,
                "error: --lda/--ldb/--ldc smaller than the shape dimension\n");
        return false;
    }
    return need(c.shape_mn, c.m_blk, c.m_max, "m")
            && need(c.shape_mn, c.n_blk, c.n_max, "n")
            && need(c.shape_k, c.k_blk, c.k_max, "k");
}

/// Calls the kernel for the whole problem, or once per M/N chunk of at most
/// `m_max`/`n_max` when M/N are `less_eq`, and once per K block of `kblock`
///
/// @note A K-blocked sequence applies the user's beta in its first call and
///     accumulates (beta 1) in the later ones, which add the bias and run the
///     post-op only in the last
void run_kernel(const matmul_case_t &c, const matmul_kernel_t &kernel,
        const bytes_t &a_data, const bytes_t &b_data, bytes_t &c_data,
        const bytes_t &bias) {
    const matmul_layout_t layout {c};
    const bool chunked = c.shape_mn == source_kind_t::less_eq;
    const int64_t m_step = chunked ? c.m_max : c.m;
    const int64_t n_step = chunked ? c.n_max : c.n;
    const int64_t k_step = c.kblock > 0 ? std::min(c.kblock, c.k) : c.k;
    const int64_t a_bytes = dtype_bytes(c.dta);
    const int64_t b_bytes = dtype_bytes(c.dtb);
    const int64_t c_bytes = dtype_bytes(c.dtc);
    const int64_t bias_bytes = dtype_bytes(c.bias_dt());

    matmul_args_t args {};
    args.lda = layout.lda() * a_bytes;
    args.ldb = layout.ldb() * b_bytes;
    args.ldc = layout.ldc() * c_bytes;
    args.alpha_bits = scalar_bits(c.alpha, c.epi());
    for (int64_t k0 = 0; k0 < c.k; k0 += k_step) {
        const bool first = k0 == 0;
        const bool last = k0 + k_step >= c.k;
        args.k = std::min(k_step, c.k - k0);
        int word = c.rt_mask == -1 ? c.post : c.rt_mask;
        double beta = c.beta;
        if (c.kblock > 0) {
            word = c.post & (post_alpha | post_beta);
            if (last) word |= c.post & (post_bias | post_postop);
            if (!first) beta = 1.0;
        }
        args.rt_flags = word;
        args.beta_bits = scalar_bits(beta, c.epi());
        for (int64_t n0 = 0; n0 < c.n; n0 += n_step) {
            for (int64_t m0 = 0; m0 < c.m; m0 += m_step) {
                args.m = std::min(m_step, c.m - m0);
                args.n = std::min(n_step, c.n - n0);
                args.a = a_data.data() + layout.a_at(m0, k0) * a_bytes;
                args.b = b_data.data() + layout.b_at(k0, n0) * b_bytes;
                args.c = c_data.data() + (n0 * layout.ldc() + m0) * c_bytes;
                const int64_t bias_m = c.bias_scalar ? 0 : m0;
                args.bias = bias.empty() ? nullptr
                                         : bias.data() + bias_m * bias_bytes;
                kernel.run(args);
            }
        }
    }
}

/// Builds and checks a kernel for one case, timing it under `run_mode_t::perf`;
/// prints a PASS/FAIL line and, on FAIL, a repro() invocation
result_t run_case(const matmul_case_t &c, const vpu_t &model, run_mode_t mode,
        expect_t expect) {
    if (!case_is_consistent(c)) return result_t::fail;
    if (!dtypes_supported(c)) {
        std::printf(
                "[%s] SKIPPED: the machine lacks the half-precision "
                "vector extensions\n",
                c.repro().c_str());
        return result_t::skipped;
    }

    matmul_kernel_t kernel(c, model);
    const bool jitted = kernel.create_kernel();
    if (!kernel.error().empty()) {
        std::fprintf(stderr, "[FAIL] assembler error: %s\n  repro: %s\n",
                kernel.error().c_str(), c.repro().c_str());
        return result_t::fail;
    }
    if (expect == expect_t::reject) {
        if (jitted) {
            std::fprintf(stderr,
                    "[FAIL] expected a plan rejection\n  repro: "
                    "%s\n",
                    c.repro().c_str());
            return result_t::fail;
        }
        std::printf("[%s] rejected as expected\n", c.repro().c_str());
        return result_t::pass;
    }
    if (!jitted) {
        if (expect == expect_t::any) {
            std::printf("[%s] REJECTED by the backend\n", c.repro().c_str());
            return result_t::rejected;
        }
        std::fprintf(stderr,
                "[FAIL] failed to JIT a kernel (set RVJIT_VERBOSE=2 for "
                "rvjit's own diagnostics)\n  repro: %s\n",
                c.repro().c_str());
        return result_t::fail;
    }

    if (const char *dir = std::getenv("RVJIT_BENCH_ASM_DIR"))
        dump_kernel_asm(dir, "matmul", kernel.getCode(), kernel.getSize());

    const matmul_layout_t layout {c};

    bytes_t a_data, b_data, bias;
    bytes_t c_data(
            layout.c_size() * dtype_bytes(c.dtc) + guard_bytes, pad_byte);
    std::vector<double> c_ref(static_cast<size_t>(c.m) * c.n);
    fill_and_reference(layout, a_data, b_data, c_data, bias, c_ref);

    run_kernel(c, kernel, a_data, b_data, c_data, bias);

    std::vector<double> c_got(static_cast<size_t>(c.m) * c.n);
    for (int64_t m = 0; m < c.m; ++m)
        for (int64_t n = 0; n < c.n; ++n)
            c_got[n * c.m + m]
                    = load_elem(c_data.data(), layout.c_at(m, n), c.dtc);

    const bool dump = std::getenv("RVJIT_BENCH_DUMP") != nullptr;
    const auto tol = tolerance_of(c);
    const auto cmp = compare(c_got.data(), c_ref.data(), c.m * c.n, tol.atol,
            tol.rtol, [&](int64_t idx, double got, double ref) {
        if (dump)
            std::fprintf(stderr, "  bad idx=%lld got=%.6f ref=%.6f\n",
                    static_cast<long long>(idx), got, ref);
    });
    const int64_t pad_bad = count_padding_writes(layout, c_data);
    const bool pass = cmp.passed() && pad_bad == 0;

    std::printf(
            "[%s] M=%lld N=%lld K=%lld dt=%s:%s:%s transA=%d transB=%d "
            "shape=%s kshape=%s ld=%s: %s (max_abs_err=%.3e max_rel_err=%.3e, "
            "%lld/%lld elements "
            "out of tolerance, %lld padding writes)\n",
            c.name.empty() ? "case" : c.name.c_str(),
            static_cast<long long>(c.m), static_cast<long long>(c.n),
            static_cast<long long>(c.k), dtype_name(c.dta), dtype_name(c.dtb),
            dtype_name(c.dtc), c.trans_a, c.trans_b, to_string(c.shape_mn),
            to_string(c.shape_k), to_string(c.ld_src), pass ? "PASS" : "FAIL",
            cmp.max_abs_err, cmp.max_rel_err, static_cast<long long>(cmp.n_bad),
            static_cast<long long>(cmp.n_total),
            static_cast<long long>(pad_bad));

    if (!pass) {
        std::fprintf(stderr, "  repro: %s\n", c.repro().c_str());
        return result_t::fail;
    }

    if (mode == run_mode_t::perf) {
        const auto t = time_it([&] {
            run_kernel(c, kernel, a_data, b_data, c_data, bias);
        }, c.reps);
        const double ops = 2.0 * static_cast<double>(c.m)
                * static_cast<double>(c.n) * static_cast<double>(c.k);
        std::printf(
                "  perf: best=%.3f ms (%.3f GFLOPS)  avg=%.3f ms "
                "(%.3f GFLOPS)  over %d reps\n",
                t.best_s * 1e3, gflops(ops, t.best_s), t.avg_s * 1e3,
                gflops(ops, t.avg_s), t.reps);
    }

    return result_t::pass;
}

void usage(const char *prog) {
    std::fprintf(stderr,
            "usage: %s [--reset] [--dt=a[:b]:c,...] [--acc=<dtype>,...] "
            "[--breg=<dtype>,...] [--areg=<dtype>,...] [--epi=<dtype>,...] "
            "[--bias_dt=<dtype>,...] [--bias_kind=per_m|scalar,...] "
            "[--transA=0,1,...] "
            "[--transB=0,1,...]\n"
            "          [--shape=literal|blocked|less_eq|runtime,...]  (M and "
            "N)\n"
            "          [--kshape=literal|blocked|less_eq|runtime,...] (K)\n"
            "          [--ld=literal|runtime,...] [--size_at=reg|mem,...]\n"
            "          [--m_blk=N,...] [--n_blk=N,...] [--k_blk=N,...]\n"
            "          [--m_max=N,...] [--n_max=N,...] [--k_max=N,...]\n"
            "          [--lda=N,...] [--ldb=N,...] [--ldc=N,...]\n"
            "          [--max_int_regs=N,...] [--lmul=1|2|4|8,...] "
            "[--reps=N]\n"
            "          [--post=none|alpha+beta+bias+postop,...] "
            "[--alpha=X,...] [--beta=X,...]\n"
            "          [--scalar=literal|runtime,...] "
            "[--rt=none|literal|runtime,...] [--rtmask=N,...] "
            "[--kblock=N,...]\n          [--expect=pass|any|reject]\n"
            "          [--mode=C|P] [--batch=<file>] MxNxK[_n\"name\"] ...\n\n"
            "  Dims, --m_blk/--n_blk/--k_blk, --m_max/--n_max and the lds "
            "accept "
            "`[N]v[+-N]` expressions, `v` being the f32 elements of one "
            "vector register group at the case's LMUL (e.g. 2v+3 = two "
            "vector tiles and a tail), so a batch stays valid at every "
            "VLEN.\n"
            "  --dt takes dtypes (s8 u8 s16 u16 s32 f16 bf16 f32 f64): `x` is "
            "every operand, `x:y` is x inputs accumulating into y, `a:b:c` "
            "names A, B and C. f16 and bf16 cases are skipped on machines "
            "lacking the half-precision vector extensions.\n"
            "  --acc names the type the products accumulate in (default: "
            "C's). One twice as wide as C narrows the accumulators as they "
            "are stored (floats round to nearest even, integers saturate), "
            "and widens the previous C when beta applies; bias and the "
            "scalars take the accumulation type. Not combined with "
            "--kblock.\n"
            "  --breg names the type B has once loaded (default: its own). "
            "A wider integer extends each element as the load reads it, "
            "e.g. `--dt=s32:s8:s32 --breg=s32` multiplies s32 by s8 bytes "
            "in an s32 multiply-accumulate.\n"
            "  --areg names the type A has in its registers (default: its "
            "own), converted as it is loaded, e.g. f16 weights computed in "
            "f32, or s8 extended to s32. --epi names the type the epilogue "
            "works in (default: the accumulation's), e.g. f32 for an s32 "
            "accumulation scaled and stored to s8. --bias_dt names the bias "
            "array's type (default: the epilogue's), and --bias_kind=scalar "
            "makes it a single element for every M.\n"
            "  Each positional MxNxK spec is cross-producted against every "
            "currently-set list-valued flag. --reset restores every flag to "
            "its default. --mode=C (default) checks correctness only; "
            "--mode=P also times (not meaningful under QEMU emulation).\n"
            "  Pointers, and every non-literal size/stride, are loaded by the "
            "kernel from an argument structure. `literal`, `blocked` and "
            "`runtime` shapes run the whole matmul in one kernel call; "
            "`less_eq` makes the driver call the kernel once per M/N chunk of "
            "at most --m_max/--n_max, a `less_eq` K carrying at most --k_max "
            "per call (split by --kblock). `blocked` pins the backend's "
            "block: N and K as is, M in whole vectors. K defaults to "
            "literal. --size_at=reg "
            "(default) loads every non-literal size into a register before "
            "the backend runs, mem lets the backend read it from the "
            "argument structure.\n"
            "  --expect=pass (default) requires every case to JIT and be "
            "correct; any tolerates a clean plan rejection by the backend "
            "(an assembler error or a wrong result still fails); reject "
            "requires the rejection.\n"
            "  --lda/--ldb/--ldc are leading dimensions in elements (0 = "
            "packed); one smaller than the shape dimension is an error. C is "
            "column-major MxN; A/B follow "
            "rvv::linalg::matmul::matmul_t's native convention (see "
            "matmul_kernel.hpp).\n"
            "  Set RVJIT_VERBOSE=2 for rvjit's own failure diagnostics, "
            "RVJIT_BENCH_DUMP=1 to print every out-of-tolerance element, "
            "RVJIT_BENCH_ASM_DIR=<dir> to write each kernel as "
            "assembly.\n",
            prog);
}

/// The number of values a sweep axis takes
///
/// @param v The axis values
template <typename T>
int axis_size(const std::vector<T> &v) {
    return static_cast<int>(v.size());
}

} // namespace rvjit_bench

int main(int argc, char **argv) {
    using namespace rvjit_bench;
    using namespace rvjit;
    using namespace rvjit::rvv;

    // Line-buffered so output survives a crashing kernel
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const auto toks = expand_argv(argc, argv);
    if (toks.empty()) {
        usage(argv[0]);
        return 1;
    }

    const int64_t vlenb = read_vlenb();
    if (vlenb <= 0) {
        std::fprintf(stderr,
                "error: could not read vlenb -- is the V extension "
                "available? This binary must run on RV64 hardware or under "
                "qemu-riscv64, never directly on the build host.\n");
        return 1;
    }
    std::printf("vlen: %lld bits (vlenb=%lld bytes)\n",
            static_cast<long long>(vlenb * 8), static_cast<long long>(vlenb));

    matmul_settings_t s;
    run_mode_t mode = run_mode_t::correctness;
    int64_t n_total = 0, n_pass = 0, n_rejected = 0, n_skipped = 0;
    bool any_case = false;

    for (size_t i = 0; i < toks.size(); ++i) {
        std::string name, value;
        if (parse_flag(toks[i], name, value)) {
            bool ok = true;
            if (name == "reset") {
                s = matmul_settings_t();
            } else if (name == "mode") {
                mode = (value == "P" || value == "perf")
                        ? run_mode_t::perf
                        : run_mode_t::correctness;
            } else if (name == "dt") {
                ok = parse_dtypes_list(value, s.dt);
            } else if (name == "acc" || name == "breg" || name == "areg"
                    || name == "epi" || name == "bias_dt") {
                std::vector<dtype_t> &list = name == "acc" ? s.acc
                        : name == "breg"                   ? s.breg
                        : name == "areg"                   ? s.areg
                        : name == "epi"                    ? s.epi
                                                           : s.bias_dt;
                list.clear();
                for (const auto &v : split_csv(value)) {
                    dtype_t dt = dtype_t::none;
                    if (v != "none" && !parse_dtype(v, dt)) {
                        std::fprintf(stderr, "error: invalid dtype '%s'\n",
                                v.c_str());
                        ok = false;
                    }
                    list.push_back(dt);
                }
                ok = ok && !list.empty();
            } else if (name == "bias_kind") {
                s.bias_scalar.clear();
                for (const auto &v : split_csv(value)) {
                    if (v != "per_m" && v != "scalar") {
                        std::fprintf(stderr, "error: invalid bias kind '%s'\n",
                                v.c_str());
                        ok = false;
                    }
                    s.bias_scalar.push_back(v == "scalar");
                }
                ok = ok && !s.bias_scalar.empty();
            } else if (name == "transA") {
                s.trans_a = parse_bool_list(value);
            } else if (name == "transB") {
                s.trans_b = parse_bool_list(value);
            } else if (name == "shape") {
                ok = parse_source_list(value, s.shape, true);
            } else if (name == "kshape") {
                ok = parse_source_list(value, s.kshape, true);
            } else if (name == "ld") {
                ok = parse_source_list(value, s.ld, false);
            } else if (name == "size_at") {
                ok = parse_size_at_list(value, s.size_at);
            } else if (name == "m_blk") {
                s.m_blk = split_csv(value);
            } else if (name == "n_blk") {
                s.n_blk = split_csv(value);
            } else if (name == "k_blk") {
                s.k_blk = split_csv(value);
            } else if (name == "m_max") {
                s.m_max = split_csv(value);
            } else if (name == "n_max") {
                s.n_max = split_csv(value);
            } else if (name == "k_max") {
                s.k_max = split_csv(value);
            } else if (name == "lda") {
                s.lda = split_csv(value);
            } else if (name == "ldb") {
                s.ldb = split_csv(value);
            } else if (name == "ldc") {
                s.ldc = split_csv(value);
            } else if (name == "max_int_regs") {
                s.max_int_regs = parse_int_list(value);
            } else if (name == "lmul") {
                s.lmul = parse_int_list(value);
            } else if (name == "post") {
                ok = parse_post_list(value, s.post);
            } else if (name == "alpha") {
                s.alpha = parse_double_list(value);
            } else if (name == "beta") {
                s.beta = parse_double_list(value);
            } else if (name == "scalar") {
                ok = parse_source_list(value, s.scalar, false);
            } else if (name == "rt") {
                ok = parse_rt_list(value, s.rt);
            } else if (name == "rtmask") {
                s.rtmask = parse_int_list(value);
            } else if (name == "kblock") {
                s.kblock = parse_int_list(value);
            } else if (name == "expect") {
                if (value == "pass") {
                    s.expect = expect_t::pass;
                } else if (value == "any") {
                    s.expect = expect_t::any;
                } else if (value == "reject") {
                    s.expect = expect_t::reject;
                } else {
                    std::fprintf(stderr, "error: invalid --expect '%s'\n",
                            value.c_str());
                    ok = false;
                }
            } else if (name == "reps") {
                s.reps = std::atoll(value.c_str());
            } else if (name == "cache_size") {
                s.cache_size = std::atoll(value.c_str());
            } else if (name == "cache_line_size") {
                s.cache_line_size = std::atoll(value.c_str());
            } else if (name == "cache_latency") {
                s.cache_latency = std::atoll(value.c_str());
            } else if (name == "help") {
                usage(argv[0]);
                return 0;
            } else {
                std::fprintf(
                        stderr, "error: unknown flag --%s\n", name.c_str());
                ok = false;
            }
            if (!ok) return 1;
            continue;
        }

        std::string spec = toks[i];
        const std::string case_name = extract_name(spec);
        const auto dim_exprs = split_dims(spec);
        if (dim_exprs.size() != 3) {
            std::fprintf(stderr,
                    "error: expected an MxNxK dims spec, got '%s'\n",
                    toks[i].c_str());
            return 1;
        }
        any_case = true;

        vpu_t model;
        model.from_params(static_cast<int>(vlenb * 8),
                static_cast<int>(s.cache_size),
                static_cast<int>(s.cache_line_size),
                static_cast<int>(s.cache_latency));

        for_each_combo(
                {axis_size(s.dt), axis_size(s.trans_a), axis_size(s.trans_b),
                        axis_size(s.shape), axis_size(s.kshape),
                        axis_size(s.ld), axis_size(s.size_at),
                        axis_size(s.m_blk), axis_size(s.n_blk),
                        axis_size(s.k_blk), axis_size(s.m_max),
                        axis_size(s.n_max), axis_size(s.k_max),
                        axis_size(s.lda), axis_size(s.ldb), axis_size(s.ldc),
                        axis_size(s.max_int_regs), axis_size(s.lmul),
                        axis_size(s.post), axis_size(s.alpha),
                        axis_size(s.beta), axis_size(s.scalar), axis_size(s.rt),
                        axis_size(s.rtmask), axis_size(s.kblock),
                        axis_size(s.acc), axis_size(s.breg), axis_size(s.areg),
                        axis_size(s.epi), axis_size(s.bias_dt),
                        axis_size(s.bias_scalar)},
                [&](const std::vector<int> &idx) {
            matmul_case_t c;
            c.name = case_name;
            int d = 0;
            const dtypes_t &dts = s.dt[idx[d++]];
            c.dta = dts.a;
            c.dtb = dts.b;
            c.dtc = dts.c;
            c.trans_a = s.trans_a[idx[d++]];
            c.trans_b = s.trans_b[idx[d++]];
            c.shape_mn = s.shape[idx[d++]];
            c.shape_k = s.kshape[idx[d++]];
            c.ld_src = s.ld[idx[d++]];
            c.size_at = s.size_at[idx[d++]];
            const std::string &m_blk = s.m_blk[idx[d++]];
            const std::string &n_blk = s.n_blk[idx[d++]];
            const std::string &k_blk = s.k_blk[idx[d++]];
            const std::string &m_max = s.m_max[idx[d++]];
            const std::string &n_max = s.n_max[idx[d++]];
            const std::string &k_max = s.k_max[idx[d++]];
            const std::string &lda = s.lda[idx[d++]];
            const std::string &ldb = s.ldb[idx[d++]];
            const std::string &ldc = s.ldc[idx[d++]];
            c.max_int_regs = static_cast<int>(s.max_int_regs[idx[d++]]);
            c.lmul = static_cast<int>(s.lmul[idx[d++]]);
            c.post = s.post[idx[d++]];
            c.alpha = s.alpha[idx[d++]];
            c.beta = s.beta[idx[d++]];
            c.scalar_src = s.scalar[idx[d++]];
            c.rt = s.rt[idx[d++]];
            c.rt_mask = static_cast<int>(s.rtmask[idx[d++]]);
            c.kblock = s.kblock[idx[d++]];
            const dtype_t acc = s.acc[idx[d++]];
            c.dtacc = acc == dtype_t::none ? c.dtc : acc;
            c.dtbreg = s.breg[idx[d++]];
            c.dtareg = s.areg[idx[d++]];
            c.dtepi = s.epi[idx[d++]];
            c.dtbias = s.bias_dt[idx[d++]];
            c.bias_scalar = s.bias_scalar[idx[d++]];
            c.reps = static_cast<int>(s.reps);

            // `v` is the A elements one input vector register group holds,
            // at the LMUL the backend will use (4, or 2 widening, unless
            // forced or capped)
            const int default_group = c.is_widening() ? 2 : 4;
            const int64_t group = c.lmul > 0
                    ? c.lmul
                    : std::min(default_group, vgroup_size(model.max_lmul));
            const int64_t vl = model.vlenb * group / dtype_bytes(c.areg());
            const auto eval = [&](const std::string &e) {
                const int64_t v = eval_vl_expr(e, vl);
                if (v < 0) {
                    std::fprintf(stderr, "error: bad size expression '%s'\n",
                            e.c_str());
                    std::exit(1);
                }
                return v;
            };
            c.vl = static_cast<int>(vl);
            c.m = eval(dim_exprs[0]);
            c.n = eval(dim_exprs[1]);
            c.k = eval(dim_exprs[2]);
            c.m_blk = static_cast<int>(eval(m_blk));
            c.n_blk = static_cast<int>(eval(n_blk));
            c.k_blk = static_cast<int>(eval(k_blk));
            c.m_max = static_cast<int>(eval(m_max));
            c.n_max = static_cast<int>(eval(n_max));
            c.k_max = static_cast<int>(eval(k_max));
            c.lda = eval(lda);
            c.ldb = eval(ldb);
            c.ldc = eval(ldc);

            ++n_total;
            switch (run_case(c, model, mode, s.expect)) {
                case result_t::pass: ++n_pass; break;
                case result_t::rejected: ++n_rejected; break;
                case result_t::skipped: ++n_skipped; break;
                case result_t::fail: break;
            }
        });
    }

    if (!any_case) {
        usage(argv[0]);
        return 1;
    }

    std::printf(
            "%lld/%lld cases passed (%lld rejected by the backend, %lld "
            "skipped)\n",
            static_cast<long long>(n_pass), static_cast<long long>(n_total),
            static_cast<long long>(n_rejected),
            static_cast<long long>(n_skipped));
    return n_pass + n_rejected + n_skipped == n_total ? 0 : 1;
}
