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
#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64/stream/iterator/iterator_helpers.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;

// JIT-executed: single pivot axis, sweeping the register-minimization cases

/// Builds a probe walking a single pivot axis
///
/// @details The probe captures:
///     - every resolved address
///     - the final base
///
/// @note A large n (multi-strip case c) unrolls thousands of instructions,
///     past the default 4KB code buffer, so callers use a 1<<20 buffer
/// @note `bias` receives the plan's `origin_bias()` at generation
rv64_jit_probe_t::generate_fn_t iterator_stream_pivot_probe(
        int n, int64_t stride_val, bool forward, int *bias) {
    return [n, stride_val, forward, bias](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_final_ptr = a2;
        const Reg breg = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(n, const_t(static_cast<int>(stride_val)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        forward ? update_t::advance(0) : update_t::preserve()),
                env.pool.integers().available());
        *bias = plan.origin_bias();
        env.iterators.new_iterator(plan);
        auto *stream = env.iterators.iterator(plan);

        // init() for every position must complete entirely before the
        // first produce() call -- never interleaved with it (see
        // pivot_t::is_eager()'s own @pre)
        index_t idx {};
        for (int j = 0; j < n; ++j) {
            idx[0] = j;
            stream->init_iteration(idx);
        }
        for (int j = 0; j < n; ++j) {
            idx[0] = j;
            const pointer_t p = stream->pointer(idx);
            store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                    static_cast<int64_t>(j) * 8);
            stream->iteration(idx);
        }

        gen.sd(breg, out_final_ptr, 0);
        gen.ret();
    };
}

class iterator_stream_pivot_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t, bool>> {};

/// A pivot axis walks base+j*stride in every case and lands its tail step,
/// an advancing origin past its span by the plan's bias
TEST_P(iterator_stream_pivot_t, addresses_and_tail_landing) {
    int n;
    int64_t stride;
    bool forward;
    std::tie(n, stride, forward) = GetParam();

    static const int64_t base_value = 100000;

    int bias = 0;
    rv64_jit_probe_t kernel(
            1 << 20, iterator_stream_pivot_probe(n, stride, forward, &bias));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(n, 0);
    int64_t out_final_base = 0;
    kernel(base_value, out_addrs.data(), &out_final_base);

    expect_address_progression(out_addrs, base_value, stride);

    const int64_t expected_final
            = forward ? base_value + n * stride + bias : base_value;
    ASSERT_EQ(out_final_base, expected_final);
}

/// Case (a)/(b)/(c) register-minimization boundaries, forward/reverse
INSTANTIATE_TEST_SUITE_P(Cases, iterator_stream_pivot_t,
        ::testing::Values(
                // Case (a): fits with no register at all
                std::make_tuple(64, 8, false), std::make_tuple(64, 8, true),
                // Case (a)/(b) boundary: span just past 2047 vs within it
                std::make_tuple(256, 8, false), // span=255*8=2040 (a)
                std::make_tuple(257, 8, false), // span=256*8=2048 (b)
                // Case (b): one shifted register
                std::make_tuple(300, 8, false), std::make_tuple(300, 8, true),
                std::make_tuple(300, -8, true),
                // Case (c): multiple strips, forces an inter-strip stride
                // materialize (borrowed scratch, see PivotCaseCMultiStrip)
                std::make_tuple(2000, 8, false),
                std::make_tuple(2000, 8, true)));

/// Builds a probe walking a 2-axis base_only pivot chain in two phases
///
/// @details The phases, in order:
///     - a full ascending init() sweep across every position
///     - a separate full main sweep
rv64_jit_probe_t::generate_fn_t multibase_chain_probe(
        int rows, int cols, int64_t row_stride, int64_t col_stride) {
    return [rows, cols, row_stride, col_stride](
                   rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_relanded_ptr = a2;
        const Reg breg = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(rows, const_t(static_cast<int>(row_stride)))
                .axis(cols, const_t(static_cast<int>(col_stride)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_only,
                        update_t::advance(0)),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        // A large chain reaches into the callee-saved s-registers once its
        // combined register cost exceeds the caller-saved budget
        env.stack.preserve();
        auto *stream = env.iterators.iterator(plan);

        index_t idx {};
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                idx[0] = r;
                idx[1] = c;
                stream->init_iteration(idx);
            }

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                idx[0] = r;
                idx[1] = c;
                const pointer_t p = stream->pointer(idx);
                store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                        static_cast<int64_t>(r * cols + c) * 8);
                stream->iteration(idx);
            }
        }

        // Re-reading {0, 0} with no further init() proves the chain's
        // whole-space end-state landing (the existing pivot_t::delegates()
        // mechanism, axis 0 delegating its own advance into the
        // chain owner's registers) moved the right registers, batched,
        // exactly once
        idx[0] = 0;
        idx[1] = 0;
        const pointer_t relanded = stream->pointer(idx);
        store_pointer(gen, relanded, value_tmp, addr_tmp, out_relanded_ptr, 0);

        env.stack.restore();
        gen.ret();
    };
}

class iterator_stream_multibase_chain_t
    : public ::testing::TestWithParam<std::tuple<int, int, int64_t, int64_t>> {
};

/// A 2-axis base_only pivot chain walks base+r*row+c*col after an init sweep
TEST_P(iterator_stream_multibase_chain_t, addresses_and_relanding) {
    int rows, cols;
    int64_t row_stride, col_stride;
    std::tie(rows, cols, row_stride, col_stride) = GetParam();

    static const int64_t base_value = 100000;

    rv64_jit_probe_t kernel(
            1 << 20, multibase_chain_probe(rows, cols, row_stride, col_stride));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(rows * cols, 0);
    int64_t out_relanded = 0;
    kernel(base_value, out_addrs.data(), &out_relanded);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int64_t want = base_value + r * row_stride + c * col_stride;
            const int64_t got = out_addrs[r * cols + c];
            ASSERT_EQ(got, want) << "at r=" << r << " c=" << c;
        }

    ASSERT_EQ(out_relanded, base_value + rows * row_stride);
}

/// A 2x2 shape within simm12, and a larger one that overflows into neighbors
INSTANTIATE_TEST_SUITE_P(Cases, iterator_stream_multibase_chain_t,
        ::testing::Values(
                std::make_tuple(2, 2, 8, 4), std::make_tuple(3, 3, 2000, 8)));

class iterator_pivot_columns_t : public ::testing::TestWithParam<bool> {};

/// Column registers under a case-a outer axis are built once, each at its
/// first position, from an immediate (strips) or a register stride, the first
/// being the advancing origin, and each steps right after its last first-row
/// read
TEST_P(iterator_pivot_columns_t, built_once_each_at_its_first_position) {
    const bool runtime = GetParam();
    static const int64_t base_value = 100000;
    static const int rows = 8, cols = 6;
    static const int row_stride = 4, col_stride = 1024;

    int nregs = 0, per_pivot = 0, bias = 0;
    std::vector<int> emitting; // `r * cols + c` of each init that emitted
    std::vector<int> stepping; // `r * cols + c` of each step that emitted
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5, a6}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_relanded_ptr = a2;
        const Reg stride_in = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;
        const Reg breg = a6;

        gen.mv(breg, base_value_in);

        const const_t col = runtime ? const_t(stride_in) : const_t(col_stride);
        space_t space;
        space.axis(rows, const_t(row_stride)).axis(cols, col);

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::advance(0)),
                env.pool.integers().available());
        nregs = plan.axis(1).nregs();
        per_pivot = plan.axis(1).positions_per_pivot;
        bias = plan.origin_bias();
        env.iterators.new_iterator(plan);
        auto *stream = env.iterators.iterator(plan);

        index_t idx {};
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                idx[0] = r;
                idx[1] = c;
                const size_t before = gen.getSize();
                stream->init_iteration(idx);
                if (gen.getSize() != before) emitting.push_back(r * cols + c);
            }

        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                idx[0] = r;
                idx[1] = c;
                const pointer_t p = stream->pointer(idx);
                store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                        static_cast<int64_t>(r * cols + c) * 8);
                const size_t before = gen.getSize();
                stream->iteration(idx);
                if (gen.getSize() != before) stepping.push_back(r * cols + c);
            }

        // The outer advance lands on the column registers, so {0, 0} reads
        // one row on
        idx[0] = 0;
        idx[1] = 0;
        const pointer_t relanded = stream->pointer(idx);
        store_pointer(gen, relanded, value_tmp, addr_tmp, out_relanded_ptr, 0);
        gen.sd(breg, out_relanded_ptr, 8);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_GE(nregs, 2);

    // A register stride has one register, so one position, per column; its
    // first column is the origin itself, built by nothing
    const int positions = runtime ? 1 : per_pivot;
    std::vector<int> want_emitting;
    for (int s = runtime ? 1 : 0; s < nregs; ++s)
        want_emitting.push_back(s * positions);
    EXPECT_EQ(emitting, want_emitting);

    // Each register steps at its last column, the last register at the row's
    std::vector<int> want_stepping;
    for (int s = 0; s < nregs; ++s)
        want_stepping.push_back(std::min((s + 1) * positions, cols) - 1);
    EXPECT_EQ(stepping, want_stepping);

    std::vector<int64_t> out_addrs(rows * cols, 0);
    int64_t out_relanded[2] = {};
    const int64_t stride = col_stride;
    kernel(base_value, out_addrs.data(), out_relanded, stride);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int64_t want = base_value + r * row_stride + c * col_stride;
            const int64_t got = out_addrs[r * cols + c];
            ASSERT_EQ(got, want) << "at r=" << r << " c=" << c;
        }
    EXPECT_EQ(out_relanded[0], base_value + rows * row_stride);
    EXPECT_EQ(out_relanded[1], base_value + rows * row_stride + bias);
}

/// Immediate and register column strides
INSTANTIATE_TEST_SUITE_P(
        strides, iterator_pivot_columns_t, ::testing::Values(false, true));

/// A register chain lands with one add per position read, across a resize
TEST(iterator_pivot_register_chain, advances_one_add_per_read_position) {
    static const int64_t base_value = 100000;
    static const int64_t stride = 3000;
    static const int full = 8, narrow = 5;

    std::vector<size_t> advance_bytes;
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5, a6}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_relanded_ptr = a2;
        const Reg stride_in = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;
        const Reg breg = a6;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(full, const_t(stride_in));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::advance(0)),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        env.stack.preserve();
        auto *stream = env.iterators.iterator(plan);

        index_t idx {};
        for (int j = 0; j < full; ++j) {
            idx[0] = j;
            stream->init_iteration(idx);
        }

        // A full pass, then a narrowed one reading on from where it landed
        int slot = 0;
        for (int size : {full, narrow}) {
            index_t shape {};
            shape[0] = size;
            stream->resize(shape);
            for (int j = 0; j < size; ++j) {
                idx[0] = j;
                const pointer_t p = stream->pointer(idx);
                store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                        static_cast<int64_t>(slot++) * 8);
                const size_t before = gen.getSize();
                stream->iteration(idx);
                if (j == size - 1)
                    advance_bytes.push_back(gen.getSize() - before);
            }
        }

        idx[0] = 0;
        const pointer_t relanded = stream->pointer(idx);
        store_pointer(gen, relanded, value_tmp, addr_tmp, out_relanded_ptr, 0);
        env.stack.restore();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    // Each pass's landing is one 4-byte add per position it read
    const std::vector<size_t> want_bytes {full * 4, narrow * 4};
    EXPECT_EQ(advance_bytes, want_bytes);

    std::vector<int64_t> out_addrs(full + narrow, 0);
    int64_t out_relanded = 0;
    kernel(base_value, out_addrs.data(), &out_relanded, stride);
    expect_address_progression(out_addrs, base_value, stride);
    EXPECT_EQ(out_relanded, base_value + (full + narrow) * stride);
}

// JIT-executed: automatic delegate advance

/// The delegate advance gives the same addresses as a fresh origin+k*ks+n*ns
TEST(iterator_delegate_advance, automatic_advance_across_rows) {
    static const int64_t base_value = 100000;
    static const int kb = 2;
    static const int n = 10;
    // kb * k_stride must stay within simm12 for K to resolve to case-a (0
    // registers) -- otherwise K would need its own chain, sidestepping the
    // delegate mechanism this test exercises
    static const int64_t k_stride = 800;
    // (n - 1) * n_stride must exceed simm12 for N to force a real (case
    // b/c) chain -- the delegate this test exercises
    static const int64_t n_stride = 300;

    // Byte size of the codegen buffer captured right after every (k, n)
    // position's own produce().iteration() call, so the test can confirm
    // *where* the delegate's whole-span step landed without disassembling
    // anything: eagerly (pivot_t::is_eager(), since K is axis 0 and
    // case-a), it's folded into the very last position of row 0 (k=0,
    // n=n_-1) -- that call emits visibly more bytes than an ordinary
    // interior position's call, and nothing at the sweep's true last
    // position (k=kb_-1, n=n_-1) does
    std::vector<size_t> sizes_after;

    rv64_jit_probe_t kernel(
            1 << 20, [&sizes_after](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_row2_ptr = a2;
        const Reg breg = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;

        gen.mv(breg, base_value_in);

        const int kb_ = 2;
        const int n_ = 10;
        const int64_t k_stride_ = 800;
        const int64_t n_stride_ = 300;

        space_t space;
        space.axis(kb_, const_t(static_cast<int>(k_stride_)))
                .axis(n_, const_t(static_cast<int>(n_stride_)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::advance(0)),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        auto *stream = env.iterators.iterator(plan);

        index_t idx {};
        for (int n = 0; n < n_; ++n) {
            idx[0] = 0;
            idx[1] = n;
            stream->init_iteration(idx);
        }

        for (int k = 0; k < kb_; ++k) {
            for (int n = 0; n < n_; ++n) {
                idx[0] = k;
                idx[1] = n;
                const pointer_t p = stream->pointer(idx);
                store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                        static_cast<int64_t>(k * n_ + n) * 8);
                stream->iteration(idx);
                sizes_after.push_back(gen.getSize());
            }
        }

        // The batched advance, fired when `.iteration()` above reached the
        // sweep's true last cell (k=kb_-1, n=n_-1), should have already
        // moved the delegate chain to "row kb_" -- re-reading {0, 0} (same
        // symbolic query as row 0's very first read) proves it, with no
        // init() call in between.
        idx[0] = 0;
        idx[1] = 0;
        const pointer_t row_kb = stream->pointer(idx);
        store_pointer(gen, row_kb, value_tmp, addr_tmp, out_row2_ptr, 0);

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(kb * n, 0);
    int64_t out_row_kb = 0;
    kernel(base_value, out_addrs.data(), &out_row_kb);

    for (int k = 0; k < kb; ++k)
        for (int col = 0; col < n; ++col) {
            const int64_t want = base_value + k * k_stride + col * n_stride;
            const int64_t got = out_addrs[k * n + col];
            ASSERT_EQ(got, want) << "at k=" << k << " n=" << col;
        }

    ASSERT_EQ(out_row_kb, base_value + kb * k_stride);

    // Eager timing check: the delegate's whole-span step is folded into
    // row 0's own last position (k=0, n=n_-1) instead of the sweep's true
    // last position (k=kb_-1, n=n_-1) -- so that call alone should have
    // emitted visibly more bytes than an ordinary interior position's call
    // (row 0's own second-to-last position), and the sweep's true last
    // position's own call should look just like an ordinary interior one
    ASSERT_EQ(sizes_after.size(), static_cast<size_t>(kb * n));
    const auto delta = [&](int i) {
        return sizes_after[i] - (i == 0 ? 0 : sizes_after[i - 1]);
    };
    const auto interior_delta = delta(n - 3); // row 0, an ordinary position
    // Row 0's last position carries the eager delegate step
    EXPECT_GT(delta(n - 1), interior_delta);
    // The sweep's last position looks ordinary: the step fired earlier
    EXPECT_EQ(delta(kb * n - 1), interior_delta);
}

} // namespace rvjit
