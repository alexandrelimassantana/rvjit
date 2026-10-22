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
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvv/stream/memory/memory_operand_helpers.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;
using rv64_iterator_t = rvjit::rv64::stream::iterator::iterator_t;
using memory_operand_t = rvjit::stream::memory_operand::memory_operand_t;
using memory_spec_t = rvjit::stream::memory_operand::spec_t;
using memory_variant_t = rvjit::stream::memory_operand::variant_t;

// JIT-executed: 1D scalar load+store round trip via iteration()

/// Builds a probe that loads `n` scalar values through one memory operand
/// and stores them through another, sharing one ring-buffer register
rv64_jit_probe_t::generate_fn_t scalar_roundtrip_probe(
        int n, type_t dt, bool is_float) {
    return [n, dt, is_float](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1}));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;
        const int stride = sewb(dt);

        space_t in_space, out_space;
        in_space.axis(n, const_t(stride));
        out_space.axis(n, const_t(stride));

        memory::op_t load_op = memory::op_t().scalar().load().as_type(dt);
        memory::op_t store_op = load_op;
        store_op.store();

        // Load and store streams share one physical register: a load
        // followed immediately by a store of the same ring slot is a pure
        // pass-through, no explicit mv needed.
        const name_t id = is_float ? env.pool.floats().pop()
                                   : env.pool.integers().pop();
        const buffer::buffer_t regs(&id, 1);
        const buffer_map_t ring(regs, {1, 1});

        auto *load_stream = env.memops.create(
                make_spec(load_op, pointer_t(in_ptr, 0), in_space),
                make_variant(rv64::stream::iterator::guided, 0, ring));
        auto *store_stream = env.memops.create(
                make_spec(store_op, pointer_t(out_ptr, 0), out_space),
                make_variant(rv64::stream::iterator::guided, 0, ring));
        ASSERT_TRUE(load_stream->is_valid());
        ASSERT_TRUE(store_stream->is_valid());

        for (int i = 0; i < n; ++i) {
            load_stream->iteration({i});
            store_stream->iteration({i});
        }
        gen.ret();
    };
}

class memory_operand_scalar_roundtrip_t
    : public ::testing::TestWithParam<std::tuple<int, type_t, bool>> {};

/// A scalar load/store round trip reproduces the input for int and float dtypes
TEST_P(memory_operand_scalar_roundtrip_t, values) {
    int n;
    type_t dt;
    bool is_float;
    std::tie(n, dt, is_float) = GetParam();

    rv64_jit_probe_t kernel(scalar_roundtrip_probe(n, dt, is_float));
    ASSERT_TRUE(kernel.create_kernel());

    if (is_float) {
        std::vector<double> in(n), out(n, 0.0);
        for (int i = 0; i < n; ++i)
            in[i] = 1.5 * (i + 1);
        kernel(in.data(), out.data());
        for (int i = 0; i < n; ++i)
            ASSERT_EQ(out[i], in[i]) << "at i=" << i;
    } else {
        std::vector<int64_t> in(n), out(n, 0);
        for (int i = 0; i < n; ++i)
            in[i] = 1000 + i;
        kernel(in.data(), out.data());
        for (int i = 0; i < n; ++i)
            ASSERT_EQ(out[i], in[i]) << "at i=" << i;
    }
}

/// Int and float dtypes, single and multi-element sizes
INSTANTIATE_TEST_SUITE_P(Kinds, memory_operand_scalar_roundtrip_t,
        ::testing::Values(std::make_tuple(5, type_t::s64, false),
                std::make_tuple(5, type_t::f64, true),
                std::make_tuple(1, type_t::s64, false)));

// JIT-executed: 2D pivot-strategy scalar load, the pattern matmul's B
// addressing relies on

/// Builds a probe walking a 2D pivot-strategy scalar load, ascending
rv64_jit_probe_t::generate_fn_t pivot_2d_probe(
        int rows, int cols, int64_t row_stride, int64_t col_stride) {
    return [rows, cols, row_stride, col_stride](
                   rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1}));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;

        space_t space;
        space.axis(rows, const_t(static_cast<int>(row_stride)))
                .axis(cols, const_t(static_cast<int>(col_stride)));
        memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

        auto *stream = env.memops.create(
                make_spec(op, pointer_t(in_ptr, 0), space),
                make_variant(rv64::stream::iterator::pivot,
                        env.pool.integers().available(), buffer_map_t({1, 1})));
        ASSERT_TRUE(stream->is_valid());

        // init() for every position must complete entirely before the
        // first produce() call -- never interleaved with it (see
        // pivot_t::is_eager()'s own @pre)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                stream->init_iteration({r, c});

        // Writes walk out_ptr forward by a constant 8-byte step (rather
        // than a raw `k*8` immediate, which would overflow simm12 for a
        // large `cols`), independent of the pivot register-count case
        // under test.
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                stream->iteration({r, c});
                rv64::memory::store_t(env.e).xstore(
                        Reg(stream->get({r, c})), out_ptr, 8);
                if (!(r == rows - 1 && c == cols - 1))
                    gen.addi(out_ptr, out_ptr, 8);
            }
        }
        gen.ret();
    };
}

class memory_operand_pivot_2d_t
    : public ::testing::TestWithParam<std::tuple<int, int, int64_t, int64_t>> {
};

/// A 2D pivot-strategy load reproduces the input matrix exactly
TEST_P(memory_operand_pivot_2d_t, values_match) {
    int rows, cols;
    int64_t row_stride, col_stride;
    std::tie(rows, cols, row_stride, col_stride) = GetParam();

    ASSERT_EQ(row_stride % 8, 0);
    ASSERT_EQ(col_stride % 8, 0);
    const int64_t max_off = (rows - 1) * row_stride + (cols - 1) * col_stride;
    std::vector<int64_t> in(max_off / 8 + 1, 0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            in[(r * row_stride + c * col_stride) / 8]
                    = static_cast<int64_t>(r) * 1000 + c;

    rv64_jit_probe_t kernel(
            1 << 20, pivot_2d_probe(rows, cols, row_stride, col_stride));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out(rows * cols, 0);
    kernel(in.data(), out.data());

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const auto got = out[r * cols + c];
            const auto want = in[(r * row_stride + c * col_stride) / 8];
            ASSERT_EQ(got, want) << "at r=" << r << " c=" << c;
        }
}

/// Small shapes (0-register case a) and large ones (multi-register chains)
INSTANTIATE_TEST_SUITE_P(Cases, memory_operand_pivot_2d_t,
        ::testing::Values(
                // Small: case (a), 0 registers on both axes
                std::make_tuple(4, 6, 64, 8),
                // Large: forces multi-register pivot chains
                std::make_tuple(3, 300, 8, 8)));

// JIT-executed: resize without a fresh init() -- the exact
// init-once/resize-then-produce pattern loop::reduce_t drives in production
// (common/loop/reduce.hpp's per-K-stage `rs1.resize({u, n1})`)

/// Builds a probe that inits a {K, N} pivot-addressed scalar-load memory
/// operand at K's full width once, then resizes K narrower via
/// resize() in place and drives iteration()/get() only
rv64_jit_probe_t::generate_fn_t memory_operand_resize_without_reinit_probe(
        int k_full, int k_narrow, int ncols, int64_t k_stride,
        int64_t n_stride) {
    return [k_full, k_narrow, ncols, k_stride, n_stride](
                   rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1}));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;

        space_t space;
        space.axis(k_full, const_t(static_cast<int>(k_stride)))
                .axis(ncols, const_t(static_cast<int>(n_stride)));
        memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

        // A single K-row ring, reused every K position -- the same small,
        // wraparound shape matmul's own B ring uses at K-ring depth 1
        auto *full_stream
                = env.memops.create(make_spec(op, pointer_t(in_ptr, 0), space),
                        make_variant(rv64::stream::iterator::pivot,
                                env.pool.integers().available(),
                                buffer_map_t({1, ncols})));
        ASSERT_TRUE(full_stream->is_valid());

        for (int k = 0; k < k_full; ++k)
            for (int n = 0; n < ncols; ++n)
                full_stream->init_iteration({k, n});

        // Resize K narrower -- same pointer registers/window, smaller
        // shape -- and drive produce()/get() only, with no further init()
        full_stream->resize({k_narrow, ncols});
        auto *narrow_stream = full_stream;

        for (int k = 0; k < k_narrow; ++k) {
            for (int n = 0; n < ncols; ++n) {
                narrow_stream->iteration({k, n});
                rv64::memory::store_t(env.e).xstore(
                        Reg(narrow_stream->get({k, n})), out_ptr, 8);
                if (!(k == k_narrow - 1 && n == ncols - 1))
                    gen.addi(out_ptr, out_ptr, 8);
            }
        }
        gen.ret();
    };
}

class memory_operand_resize_without_reinit_t
    : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

/// A narrowed {K, N} operand matches a narrow build without a new init
TEST_P(memory_operand_resize_without_reinit_t,
        produces_without_reinit_after_resize) {
    int k_full, k_narrow, ncols;
    std::tie(k_full, k_narrow, ncols) = GetParam();
    static const int64_t k_stride = 304; // multi-strip: 19*304=5776 > 2047
    static const int64_t n_stride = 8;

    rv64_jit_probe_t kernel(1 << 20,
            memory_operand_resize_without_reinit_probe(
                    k_full, k_narrow, ncols, k_stride, n_stride));
    ASSERT_TRUE(kernel.create_kernel());

    const int64_t max_off = (k_full - 1) * k_stride + (ncols - 1) * n_stride;
    std::vector<int64_t> in(max_off / 8 + 1, 0);
    for (int k = 0; k < k_full; ++k)
        for (int n = 0; n < ncols; ++n)
            in[(k * k_stride + n * n_stride) / 8]
                    = static_cast<int64_t>(k) * 1000 + n;

    std::vector<int64_t> out(k_narrow * ncols, 0);
    kernel(in.data(), out.data());

    for (int k = 0; k < k_narrow; ++k)
        for (int n = 0; n < ncols; ++n) {
            const auto got = out[k * ncols + n];
            const auto want = in[(k * k_stride + n * n_stride) / 8];
            ASSERT_EQ(got, want) << "at k=" << k << " n=" << n;
        }
}

/// A K width within the first pivot strip, and one reaching the second
INSTANTIATE_TEST_SUITE_P(Cases, memory_operand_resize_without_reinit_t,
        ::testing::Values(
                std::make_tuple(20, 10, 3), std::make_tuple(20, 15, 3)));

// JIT-executed: ring-buffer wraparound (ring_shape smaller than axis size)

/// A ring smaller than the axis keeps values read before their slot is reused
TEST(memory_operand_ring, wraparound_preserves_values_read_before_reuse) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1}));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;

        space_t space;
        space.axis(5, const_t(8));
        memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

        // Only a 2-deep ring backs 5 logical reads
        auto *stream
                = env.memops.create(make_spec(op, pointer_t(in_ptr, 0), space),
                        make_variant(rv64::stream::iterator::guided, 0,
                                buffer_map_t({2, 1})));
        ASSERT_TRUE(stream->is_valid());

        for (int i = 0; i < 5; ++i) {
            stream->iteration({i});
            // Consumed immediately, before this ring slot is reused 2
            // iterations later
            rv64::memory::store_t(env.e).xstore(
                    Reg(stream->get({i})), pointer_t(out_ptr, i * 8), 8);
        }
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> in = {10, 20, 30, 40, 50};
    std::vector<int64_t> out(5, 0);
    kernel(in.data(), out.data());

    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(out[i], in[i]) << "at i=" << i;
}

/// A pool-backed memory operand loads its values correctly end to end
TEST(memory_operand_auto, end_to_end_pool_backed_load) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1}));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;

        space_t space;
        space.axis(4, const_t(8));
        memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

        auto *stream = env.memops.create(
                make_spec(op, pointer_t(in_ptr, 0), space),
                make_variant(rv64::stream::iterator::guided,
                        env.pool.integers().available(), buffer_map_t({2, 1})));
        ASSERT_TRUE(stream->is_valid());

        for (int i = 0; i < 4; ++i) {
            stream->iteration({i});
            rv64::memory::store_t(env.e).xstore(
                    Reg(stream->get({i})), pointer_t(out_ptr, i * 8), 8);
        }
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> in = {10, 20, 30, 40};
    std::vector<int64_t> out(4, 0);
    kernel(in.data(), out.data());

    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(out[i], in[i]) << "at i=" << i;
}

} // namespace rvjit
