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

#ifndef RVJIT_TEST_BENCH_STREAM_COMMON_HPP
#define RVJIT_TEST_BENCH_STREAM_COMMON_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bench/common/asm_dump.hpp"
#include "bench/common/cli.hpp"
#include "bench/common/csr.hpp"

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/stream/iterator/plan.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit_bench {

using namespace rvjit;
using rvjit::stream::iterator::plan_t;
using rvjit::stream::iterator::strategy_id_t;
namespace iter_rv64 = rvjit::rv64::stream::iterator;
using rvjit::stream::iterator::addressing_t;
using rvjit::stream::iterator::update_t;

/// One axis's `--stride` value: either a plain compile-time immediate, or
/// a request to materialize `value` into a register instead (`reg(N)` on
/// the command line, or automatic fallback for an immediate that doesn't
/// fit simm12 -- see `parse_stream_config`)
struct stride_slot_t {
    int64_t value = -1; // -1: unset, default to a contiguous row-major layout
    bool from_reg = false;
};

/// Caller-supplied parameters constructing an iterator/memory operand stream
/// under test -- everything a `plan`/`plan_nd`
/// request needs, plus the element type used to read the data buffer and
/// report results
///
/// @note Fixed at ND=2, matching NDIMS -- the dimensionality every other
///     stream/iterator consumer in this codebase defaults to
struct stream_config_t {
    index_t shape {8, 4};
    stride_slot_t stride[2];
    strategy_id_t strategy_hint = strategy_id_t::any();
    addressing_t addressing = addressing_t::base_simm;
    update_t end = update_t::advance(0);
    type_t dt = type_t::s32;
    int ntmp = 8;
    std::string dump_asm_dir;
};

inline strategy_id_t parse_strategy(const std::string &s) {
    if (s == "guided") return iter_rv64::guided;
    if (s == "pivot") return iter_rv64::pivot;
    return strategy_id_t::any();
}

inline addressing_t parse_addressing(const std::string &s) {
    return s == "base_only" ? addressing_t::base_only : addressing_t::base_simm;
}

/// Parses an `--end=` value of the form `kind[:axis]` (e.g. `advance:0`,
/// `preserve`)
inline update_t parse_end(const std::string &s) {
    const auto colon = s.find(':');
    const std::string kind = s.substr(0, colon);
    const int axis = colon == std::string::npos
            ? 0
            : std::atoi(s.substr(colon + 1).c_str());
    if (kind == "preserve") return update_t::preserve();
    return update_t::advance(axis);
}

/// Parses one `--stride` axis token: a plain byte count, or `reg(N)` to
/// request `N` be materialized into a register instead of baked as an
/// immediate
inline stride_slot_t parse_stride_token(const std::string &tok) {
    stride_slot_t s;
    if (tok.size() > 5 && tok.compare(0, 4, "reg(") == 0 && tok.back() == ')') {
        s.value = std::atoll(tok.substr(4, tok.size() - 5).c_str());
        s.from_reg = true;
    } else {
        s.value = std::atoll(tok.c_str());
    }
    return s;
}

/// Builds a `const_t` for one `--stride` axis, materializing it into a
/// pool-allocated register via `li` when the slot requested `reg(N)` mode
/// -- forced regardless of whether `s.value` would also fit as a plain
/// immediate, since `const_t`'s own constructors always prefer the
/// immediate form when one fits
inline const_t stride_const(
        codegen_t &cg, buffer::buffer_pool_t &pool, const stride_slot_t &s) {
    if (!s.from_reg) return const_t(static_cast<int>(s.value));
    const Reg r = Reg(pool.integers().pop());
    cg.li(r, static_cast<int>(s.value));
    return const_t(r);
}

inline type_t parse_dt(const std::string &s) {
    if (s == "s8") return type_t::s8;
    if (s == "u8") return type_t::u8;
    if (s == "s16") return type_t::s16;
    if (s == "u16") return type_t::u16;
    if (s == "s64") return type_t::s64;
    return type_t::s32;
}

/// Fills in `stream_config_t` from a flat token list (already batch-file-
/// expanded, see common/cli.hpp); returns false on a malformed token
inline bool parse_stream_config(
        const std::vector<std::string> &toks, stream_config_t &cfg) {
    for (auto &tok : toks) {
        std::string name, value;
        if (!parse_flag(tok, name, value)) {
            std::fprintf(stderr, "error: unexpected positional token '%s'\n",
                    tok.c_str());
            return false;
        }
        if (name == "shape") {
            const auto dims = parse_dims(value);
            if (dims.size() != 2) {
                std::fprintf(stderr,
                        "error: --shape must have exactly 2 dims (e.g. "
                        "8x4)\n");
                return false;
            }
            cfg.shape = {static_cast<int>(dims[0]), static_cast<int>(dims[1])};
        } else if (name == "stride") {
            const auto vals = split_csv(value);
            if (vals.size() != 2) {
                std::fprintf(stderr,
                        "error: --stride must have exactly 2 comma-"
                        "separated byte strides (each a plain immediate or "
                        "reg(N))\n");
                return false;
            }
            cfg.stride[0] = parse_stride_token(vals[0]);
            cfg.stride[1] = parse_stride_token(vals[1]);
        } else if (name == "strategy") {
            cfg.strategy_hint = parse_strategy(value);
        } else if (name == "addressing") {
            cfg.addressing = parse_addressing(value);
        } else if (name == "end") {
            cfg.end = parse_end(value);
        } else if (name == "dt") {
            cfg.dt = parse_dt(value);
        } else if (name == "ntmp") {
            cfg.ntmp = std::atoi(value.c_str());
        } else if (name == "dump-asm") {
            cfg.dump_asm_dir = value;
        } else if (name == "help") {
            return false;
        } else {
            std::fprintf(stderr, "warning: unknown flag --%s, ignoring\n",
                    name.c_str());
        }
    }

    // A default contiguous row-major layout, once dt (element size) is known
    const int elem = sewb(cfg.dt);
    if (cfg.stride[1].value < 0) cfg.stride[1].value = elem;
    if (cfg.stride[0].value < 0)
        cfg.stride[0].value = static_cast<int64_t>(cfg.shape[1]) * elem;

    // An immediate too large for simm12 sources from a register instead of
    // failing -- an explicit reg(N) token is left as-is either way
    for (auto &s : cfg.stride) {
        if (!s.from_reg && !is_simm12(static_cast<int>(s.value)))
            s.from_reg = true;
    }
    return true;
}

inline void print_stream_usage(const char *prog) {
    std::fprintf(stderr,
            "usage: %s [--shape=8x4] [--stride=b0,b1] "
            "[--strategy=any|guided|pivot] [--addressing=base_simm|"
            "base_only] [--end=preserve|advance:D] "
            "[--dt=s8|u8|s16|u16|s32|s64] [--ntmp=N] [--dump-asm=<dir>]\n\n"
            "  --stride defaults to a contiguous row-major layout over "
            "--shape at --dt's element width. Each of the two "
            "comma-separated values may instead be reg(N) to force that "
            "axis's stride to be materialized into a register (holding N) "
            "rather than baked as an immediate -- e.g. "
            "--stride=1024,reg(4) sources the inner stride from a "
            "register and the outer from a plain immediate. A plain "
            "immediate that doesn't fit a 12-bit signed field falls back "
            "to reg(N) automatically instead of failing (e.g. "
            "--stride=2048,4 sources the outer stride from a register "
            "since 2048 isn't simm12). --dump-asm writes the "
            "generated kernel as a real .text object (see "
            "test/bench/common/asm_dump.hpp) for objdump -d inspection.\n"
            "  Must run on RV64 hardware or under qemu-riscv64.\n",
            prog);
}

// Host-side oracle: row-major addressing (base + sum(idx[d]*stride[d])) that
// does not re-derive the guided/pivot logic, whichever strategy each axis got

inline int64_t flat_index(const index_t &idx, const index_t &shape) {
    return static_cast<int64_t>(idx[0]) * shape[1] + idx[1];
}

inline int64_t expected_offset(
        const index_t &idx, const stride_slot_t stride[2]) {
    return static_cast<int64_t>(idx[0]) * stride[0].value
            + static_cast<int64_t>(idx[1]) * stride[1].value;
}

/// A small, always-fits-any-integer-width, easy-to-eyeball marker value --
/// never 0, so an out-of-bounds/wrong-address read (landing on
/// zero-initialized buffer padding) is distinguishable from a legitimate read
inline int64_t expected_value(const index_t &idx, const index_t &shape) {
    return (flat_index(idx, shape) % 100) + 1;
}

inline void poke(uint8_t *buf, int64_t offset, int width, int64_t value) {
    for (int i = 0; i < width; ++i)
        buf[offset + i] = static_cast<uint8_t>((value >> (8 * i)) & 0xff);
}

inline int64_t peek(
        const uint8_t *buf, int64_t offset, int width, bool is_signed) {
    uint64_t v = 0;
    for (int i = 0; i < width; ++i)
        v |= static_cast<uint64_t>(buf[offset + i]) << (8 * i);
    if (is_signed && width < 8) {
        const uint64_t sign_bit = uint64_t(1) << (width * 8 - 1);
        if (v & sign_bit) v |= ~((sign_bit << 1) - 1);
    }
    return static_cast<int64_t>(v);
}

/// Byte size of a data buffer covering every position's own offset
///
/// @pre Both strides are non-negative (this example doesn't size buffers
///     for backward/negative-stride traversal)
inline int64_t data_buffer_bytes(const stream_config_t &cfg) {
    return expected_offset({cfg.shape[0] - 1, cfg.shape[1] - 1}, cfg.stride)
            + sewb(cfg.dt);
}

/// Builds and JIT-compiles `kernel`, runs it over a position-encoded data
/// buffer, verifies every position's readback against the host-side
/// oracle, reports the generated code's static/dynamic instruction counts,
/// and (if `--dump-asm` was set) writes the kernel's disassembly
///
/// @details `Kernel` needs `create_kernel()`, `run(const void*, void*)`,
///     `getCode()`, `getSize()` -- the shape every rv64_jit_test_generator_t
///     -like harness in this codebase already has
///
/// @note `check_correctness=false` skips the readback verification (there
///     is nothing to verify against a kernel that never wrote its results
///     buffer, e.g. bench_iterator's own `--mode=inspection`) and just
///     reports instruction counts
template <typename Kernel>
int run_and_report(const stream_config_t &cfg, Kernel &kernel,
        const char *bin_name, bool check_correctness = true) {
    if (!kernel.create_kernel()) {
        std::fprintf(stderr,
                "error: failed to JIT the kernel (set RVJIT_VERBOSE=2 for "
                "rvjit's own diagnostics)\n");
        return 1;
    }

    if (!cfg.dump_asm_dir.empty())
        dump_kernel_asm(
                cfg.dump_asm_dir, bin_name, kernel.getCode(), kernel.getSize());

    const int width = sewb(cfg.dt);
    const bool is_signed = is_signed_int(cfg.dt);
    const int64_t n = static_cast<int64_t>(cfg.shape[0]) * cfg.shape[1];

    std::vector<uint8_t> data(data_buffer_bytes(cfg), 0);
    std::vector<uint8_t> results(n * width, 0);

    for (int i0 = 0; i0 < cfg.shape[0]; ++i0)
        for (int i1 = 0; i1 < cfg.shape[1]; ++i1) {
            const index_t idx {i0, i1};
            poke(data.data(), expected_offset(idx, cfg.stride), width,
                    expected_value(idx, cfg.shape));
        }

    const int64_t instret_0 = read_instret();
    kernel.run(data.data(), results.data());
    const int64_t instret_1 = read_instret();

    bool pass = true;
    if (check_correctness) {
        int64_t n_bad = 0;
        for (int i0 = 0; i0 < cfg.shape[0]; ++i0) {
            for (int i1 = 0; i1 < cfg.shape[1]; ++i1) {
                const index_t idx {i0, i1};
                const int64_t got = peek(results.data(),
                        flat_index(idx, cfg.shape) * width, width, is_signed);
                const int64_t want = expected_value(idx, cfg.shape);
                if (got != want) {
                    ++n_bad;
                    std::fprintf(stderr,
                            "  bad idx=(%d,%d): got=%lld want=%lld\n", i0, i1,
                            static_cast<long long>(got),
                            static_cast<long long>(want));
                }
            }
        }

        pass = n_bad == 0;
        std::printf(
                "[%s] shape=%dx%d stride=%lld,%lld strategy=%s "
                "addressing=%s: %s (%lld/%lld positions correct)\n",
                bin_name, cfg.shape[0], cfg.shape[1],
                static_cast<long long>(cfg.stride[0].value),
                static_cast<long long>(cfg.stride[1].value),
                cfg.strategy_hint == iter_rv64::guided          ? "guided"
                        : cfg.strategy_hint == iter_rv64::pivot ? "pivot"
                                                                : "any",
                cfg.addressing == addressing_t::base_only ? "base_only"
                                                          : "base_simm",
                pass ? "PASS" : "FAIL", static_cast<long long>(n - n_bad),
                static_cast<long long>(n));
    } else {
        std::printf(
                "[%s] shape=%dx%d stride=%lld,%lld strategy=%s "
                "addressing=%s: no correctness check (--mode=inspection)\n",
                bin_name, cfg.shape[0], cfg.shape[1],
                static_cast<long long>(cfg.stride[0].value),
                static_cast<long long>(cfg.stride[1].value),
                cfg.strategy_hint == iter_rv64::guided          ? "guided"
                        : cfg.strategy_hint == iter_rv64::pivot ? "pivot"
                                                                : "any",
                cfg.addressing == addressing_t::base_only ? "base_only"
                                                          : "base_simm");
    }

    const int64_t static_insns = static_cast<int64_t>(kernel.getSize()) / 4;
    std::printf("  static instructions (emitted code size / 4): %lld\n",
            static_cast<long long>(static_insns));
    if (instret_1 > instret_0) {
        std::printf("  executed instructions (instret CSR delta): %lld\n",
                static_cast<long long>(instret_1 - instret_0));
    } else {
        std::printf(
                "  executed instructions: instret CSR unavailable/zero "
                "delta (qemu-user build/CPU model may not expose Zicntr to "
                "U-mode) -- falling back to the static count above\n");
    }

    return pass ? 0 : 1;
}

} // namespace rvjit_bench

#endif
