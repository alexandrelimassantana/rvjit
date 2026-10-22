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
#include "rv64/stream/iterator/iterator_helpers.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;

// JIT-executed: 1-axis direct stream

/// Builds a probe walking a 1D direct iterator, capturing every
/// resolved address and the final base
rv64_jit_probe_t::generate_fn_t iterator_stream_1d_probe(
        int n, int64_t stride_val, bool stride_from_reg, bool iterate) {
    return [n, stride_val, stride_from_reg, iterate](
                   rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_final_ptr = a2;
        const Reg breg = a3;
        const Reg stride_reg = a4;
        const Reg iptr_tmp = a5;
        const Reg value_tmp = a6;
        const Reg addr_tmp = a7;

        gen.mv(breg, base_value_in);

        const_t stride(static_cast<int>(stride_val));
        if (stride_from_reg) {
            gen.li(stride_reg, stride_val);
            stride = const_t(stride_reg);
        }

        space_t space;
        space.axis(n, stride);

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::guided, addressing_t::base_simm,
                        iterate ? update_t::preserve() : update_t::advance(0)),
                1);
        const name_t tmp[1] = {reg_name(iptr_tmp)};
        assign_regs(plan.axis(0), tmp);
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

class iterator_stream_1d_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t, bool, bool>> {};

/// A 1D direct iterator walks base+j*stride and ends at the requested base
TEST_P(iterator_stream_1d_t, address_progression) {
    int n;
    int64_t stride;
    bool stride_from_reg, iterate;
    std::tie(n, stride, stride_from_reg, iterate) = GetParam();

    static const int64_t base_value = 100000;

    rv64_jit_probe_t kernel(
            iterator_stream_1d_probe(n, stride, stride_from_reg, iterate));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(n, 0);
    int64_t out_final_base = 0;
    kernel(base_value, out_addrs.data(), &out_final_base);

    expect_address_progression(out_addrs, base_value, stride);

    const int64_t expected_final
            = iterate ? base_value : base_value + n * stride;
    ASSERT_EQ(out_final_base, expected_final);
}

/// Sizes, signed strides, imm/reg stride sources and both end-states
INSTANTIATE_TEST_SUITE_P(Sizes, iterator_stream_1d_t,
        ::testing::Combine(::testing::Values(1, 2, 5),
                ::testing::Values<int64_t>(8, -16), ::testing::Bool(),
                ::testing::Bool()));

// JIT-executed: 2-axis direct/direct space (raster tile)

/// Builds a probe walking a 2D direct/direct raster tile, capturing every
/// resolved address and the final row base
rv64_jit_probe_t::generate_fn_t iterator_stream_2d_probe(int rows, int cols,
        int64_t row_stride, int64_t col_stride, bool iterate) {
    return [rows, cols, row_stride, col_stride, iterate](
                   rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);

        const Reg base_value_in = a0;
        const Reg out_ptr = a1;
        const Reg out_final_ptr = a2;
        const Reg breg = a3;
        const Reg row_tmp = a4;
        const Reg col_tmp = a5;
        const Reg value_tmp = a6;
        const Reg addr_tmp = a7;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(rows, const_t(static_cast<int>(row_stride)))
                .axis(cols, const_t(static_cast<int>(col_stride)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::guided, addressing_t::base_simm,
                        iterate ? update_t::preserve() : update_t::advance(0)),
                2);
        const name_t tmp[2] = {reg_name(row_tmp), reg_name(col_tmp)};
        assign_regs(plan.axis(0), tmp);
        assign_regs(plan.axis(1), tmp + (plan.axis(0).ngpr > 0 ? 1 : 0));
        auto *stream = env.iterators.iterator(plan);

        // init() for every position must complete entirely before the
        // first produce() call -- never interleaved with it (see
        // pivot_t::is_eager()'s own @pre)
        index_t idx {};
        for (int r = 0; r < rows; ++r) {
            idx[0] = r;
            for (int c = 0; c < cols; ++c) {
                idx[1] = c;
                stream->init_iteration(idx);
            }
        }
        for (int r = 0; r < rows; ++r) {
            idx[0] = r;
            for (int c = 0; c < cols; ++c) {
                idx[1] = c;
                const pointer_t cell = stream->pointer(idx);
                store_pointer(gen, cell, value_tmp, addr_tmp, out_ptr,
                        static_cast<int64_t>(r * cols + c) * 8);
                stream->iteration(idx);
            }
        }

        gen.sd(breg, out_final_ptr, 0);
        gen.ret();
    };
}

class iterator_stream_2d_t
    : public ::testing::TestWithParam<
              std::tuple<int, int, int64_t, int64_t, bool>> {};

/// A 2D direct raster tile walks base+r*row_stride+c*col_stride exactly
TEST_P(iterator_stream_2d_t, row_col_addressing) {
    int rows, cols;
    int64_t row_stride, col_stride;
    bool iterate;
    std::tie(rows, cols, row_stride, col_stride, iterate) = GetParam();

    static const int64_t base_value = 100000;

    rv64_jit_probe_t kernel(iterator_stream_2d_probe(
            rows, cols, row_stride, col_stride, iterate));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(rows * cols, 0);
    int64_t out_final_base = 0;
    kernel(base_value, out_addrs.data(), &out_final_base);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int64_t want = base_value + r * row_stride + c * col_stride;
            const int64_t got = out_addrs[r * cols + c];
            ASSERT_EQ(got, want) << "at r=" << r << " c=" << c;
        }

    const int64_t expected_final
            = iterate ? base_value : base_value + rows * row_stride;
    ASSERT_EQ(out_final_base, expected_final);
}

/// Square, rectangular and negative-stride shapes; the columns walk the
/// advancing origin unless their rewind overflows the row step, as at -2048
INSTANTIATE_TEST_SUITE_P(Shapes, iterator_stream_2d_t,
        ::testing::Values(std::make_tuple(1, 1, 64, 8, false),
                std::make_tuple(3, 4, 64, 8, false),
                std::make_tuple(3, 4, -2048, 8, false),
                std::make_tuple(4, 1, 32, 8, false),
                std::make_tuple(1, 5, 64, -8, false),
                std::make_tuple(3, 4, 64, 8, true)));

// JIT-executed: resize without a fresh init() (goal ii's load-bearing
// contract -- see loop::reduce_t, common/loop/reduce.hpp)

/// Builds a probe that resizes a multi-strip pivot axis without a new init
///
/// @details The probe, in order:
///     - inits the axis once at its full width
///     - resizes it narrower in place and drives produce()/get() only
///     - re-reads position 0 with no further init()
///
/// @note The re-read confirms the narrower pass's end-of-span step, sized to
///     `n_narrow`, landed on the already-built strip registers, as
///     loop::reduce_t relies on
rv64_jit_probe_t::generate_fn_t iterator_resize_without_reinit_probe(
        int n_full, int n_narrow, int64_t stride_val) {
    return [n_full, n_narrow, stride_val](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg out_relanded_ptr = a2;
        const Reg breg = a3;
        const Reg value_tmp = a4;
        const Reg addr_tmp = a5;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(n_full, const_t(static_cast<int>(stride_val)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::advance(0)),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        auto *full_stream = env.iterators.iterator(plan);

        // init() at the full width, once, ascending
        index_t idx {};
        for (int j = 0; j < n_full; ++j) {
            idx[0] = j;
            full_stream->init_iteration(idx);
        }

        // Resize narrower -- same registers/window, smaller shape -- and
        // drive produce()/get() only, with no further init()
        full_stream->resize({n_narrow});
        auto *narrow_stream = full_stream;

        for (int j = 0; j < n_narrow; ++j) {
            idx[0] = j;
            const pointer_t p = narrow_stream->pointer(idx);
            store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                    static_cast<int64_t>(j) * 8);
            narrow_stream->iteration(idx);
        }

        idx[0] = 0;
        const pointer_t relanded = narrow_stream->pointer(idx);
        store_pointer(gen, relanded, value_tmp, addr_tmp, out_relanded_ptr, 0);

        gen.ret();
    };
}

/// A resized multi-strip pivot axis matches a narrow build, landing at n_narrow
TEST(iterator_resize_without_reinit, produces_without_reinit_after_resize) {
    static const int64_t base_value = 100000;
    static const int n_full = 20;
    static const int n_narrow = 15; // spans both of the axis's 2 strips
    static const int64_t stride = 300;

    rv64_jit_probe_t kernel(
            iterator_resize_without_reinit_probe(n_full, n_narrow, stride));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(n_narrow, 0);
    int64_t out_relanded = 0;
    kernel(base_value, out_addrs.data(), &out_relanded);

    expect_address_progression(out_addrs, base_value, stride);
    ASSERT_EQ(out_relanded, base_value + n_narrow * stride);
}

// JIT-executed: pivot resolved under `direct` addressing -- never bakes an
// offset, so every captured address must come purely from register content

/// A pivot axis under `direct` addressing never bakes a nonzero offset
TEST(iterator_addressing_round_trip,
        pivot_under_direct_addressing_never_bakes_offset) {
    static const int64_t base_value = 100000;
    // Would resolve to case (b) under `immediate`; under `direct` every
    // position needs its own register (no guided fallback -- see
    // rv64_iterator_t::plan), so n must fit the real int register
    // file (the source excludes 6 here, leaving well under 27)
    static const int n = 16;
    static const int64_t stride = 8;

    // Set during generate(): whether any resolved pointer_t along the way
    // carried a nonzero offset (which a real vector op would silently drop)
    bool saw_nonzero_offset = false;
    rv64_jit_probe_t kernel(
            1 << 20, [&saw_nonzero_offset](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg breg = a2;
        const Reg value_tmp = a3;
        const Reg addr_tmp = a4;

        gen.mv(breg, base_value_in);

        space_t space;
        space.axis(n, const_t(static_cast<int>(stride)));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space,
                        rv64::stream::iterator::pivot, addressing_t::base_only,
                        update_t::advance(0)),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        // One register per position reaches the callee-saved registers once
        // n exceeds the caller-saved budget, so preserve them
        env.stack.preserve();
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
            if (p.offset != 0) saw_nonzero_offset = true;
            store_pointer(gen, p, value_tmp, addr_tmp, out_addrs_ptr,
                    static_cast<int64_t>(j) * 8);
            stream->iteration(idx);
        }
        env.stack.restore();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_FALSE(saw_nonzero_offset);

    std::vector<int64_t> out_addrs(n, 0);
    kernel(base_value, out_addrs.data());

    expect_address_progression(out_addrs, base_value, stride);
}

// JIT-executed: rv64_iterator_t::new_iterator end-to-end

/// new_iterator() resolves a mixed direct/pivot 2D space to correct addresses
TEST(iterator_new_iterator, end_to_end) {
    static const int64_t base_value = 100000;
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2, a3, a4}));

        const Reg base_value_in = a0;
        const Reg out_addrs_ptr = a1;
        const Reg breg = a2;
        const Reg value_tmp = a3;
        const Reg addr_tmp = a4;

        gen.mv(breg, base_value_in);

        // A 2-axis space, both small enough that `any` resolves both to
        // pivot case-a (0 registers) -- exercises new_iterator() on a plan
        // that ends up needing no registers at all
        space_t space;
        space.axis(3, const_t(64)).axis(4, const_t(8));

        auto plan = rv64_iterator_t::plan(
                make_request(pointer_t(breg, 0), space, strategy_id_t::any(),
                        addressing_t::base_simm, update_t::preserve()),
                env.pool.integers().available());
        env.iterators.new_iterator(plan);
        auto *stream = env.iterators.iterator(plan);

        // init() for every position must complete entirely before the
        // first produce() call -- never interleaved with it (see
        // pivot_t::is_eager()'s own @pre)
        index_t idx {};
        for (int r = 0; r < 3; ++r) {
            idx[0] = r;
            for (int c = 0; c < 4; ++c) {
                idx[1] = c;
                stream->init_iteration(idx);
            }
        }
        for (int r = 0; r < 3; ++r) {
            idx[0] = r;
            for (int c = 0; c < 4; ++c) {
                idx[1] = c;
                const pointer_t cell = stream->pointer(idx);
                store_pointer(gen, cell, value_tmp, addr_tmp, out_addrs_ptr,
                        static_cast<int64_t>(r * 4 + c) * 8);
                stream->iteration(idx);
            }
        }
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> out_addrs(12, 0);
    kernel(base_value, out_addrs.data());

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            const int64_t want = base_value + r * 64 + c * 8;
            const int64_t got = out_addrs[r * 4 + c];
            ASSERT_EQ(got, want) << "at r=" << r << " c=" << c;
        }
}

/// new_iterator() fills the registers of every unresolved axis of a plan
TEST(iterator_new_iterator, resolves_every_axis) {
    Xbyak_riscv::CodeGenerator cg;
    model_env_t env(cg);

    space_t space;
    space.axis(3, const_t(64)).axis(2000, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(t0, 0), space,
                                              rv64::stream::iterator::pivot),
            env.pool.integers().available());
    for (int d = 0; d < plan.ndims(); ++d) {
        const auto strategy = plan.axis(d).strategy;
        ASSERT_NE(strategy, strategy_id_t::any()) << "at dim " << d;
    }

    env.iterators.new_iterator(plan);
    ASSERT_TRUE(plan.is_valid()); // now that every axis has its registers
    for (int d = 0; d < plan.ndims(); ++d) {
        {
            const auto &axis = plan.axis(d);
            EXPECT_EQ(axis.regs.size(), axis.ngpr) << "at dim " << d;
        }
        EXPECT_FALSE(plan.axis(d).needs_regs()) << "at dim " << d;
    }
}

/// new_iterator() leaves an axis needing registers when the pool can't cover it
TEST(iterator_new_iterator, pool_exhaustion_leaves_axis_unresolved) {
    Xbyak_riscv::CodeGenerator cg;
    model_env_t env(cg);

    space_t space;
    space.axis(2000, const_t(8));
    auto plan = rv64_iterator_t::plan(make_request(pointer_t(t0, 0), space,
                                              rv64::stream::iterator::pivot),
            32);
    ASSERT_NE(plan.axis(0).strategy, strategy_id_t::any());
    ASSERT_GT(plan.axis(0).ngpr, 1);

    // Exhaust the pool down to fewer registers than this axis needs.
    while (env.pool.integers().available() >= plan.axis(0).ngpr)
        Reg(env.pool.integers().pop());

    env.iterators.new_iterator(plan);
    EXPECT_TRUE(plan.axis(0).needs_regs());
}

// JIT-executed: nested_1d() composing two plain iterator streams directly,
// with no memory operand wrapper involved

/// Builds a probe that drives nested_1d() over two plain iterator streams
///
/// @details The streams and the result:
///     - A: a {K, M} row-major matrix
///     - B: a {K, N} row-major matrix
///     - stores `A[k,m] + B[k,n]` at every (k, m, n)
///
/// @note Both spaces resolve to a register-free pivot under `preserve()`, so
///     iteration() never mutates a base before `cb` reads its position
rv64_jit_probe_t::generate_fn_t nest_bare_iterator_probe(int K, int M, int N) {
    return [K, M, N](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        common::stream::combinator_t combinator;
        rvv::memory::load_t load(env.e);
        rvv::memory::store_t store(env.e);

        const Reg a_ptr = a0, b_ptr = a1, out_ptr = a2;

        space_t a_space;
        a_space.axis(K, const_t(M * 8)).axis(M, const_t(8));
        auto a_plan = env.iterators.plan(
                make_request(pointer_t(a_ptr, 0), a_space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::preserve()),
                env.pool.integers().available());
        ASSERT_TRUE(a_plan.is_valid());
        auto *a_iter = env.iterators.iterator(a_plan);

        space_t b_space;
        b_space.axis(K, const_t(N * 8)).axis(N, const_t(8));
        auto b_plan = env.iterators.plan(
                make_request(pointer_t(b_ptr, 0), b_space,
                        rv64::stream::iterator::pivot, addressing_t::base_simm,
                        update_t::preserve()),
                env.pool.integers().available());
        ASSERT_TRUE(b_plan.is_valid());
        auto *b_iter = env.iterators.iterator(b_plan);

        const Reg ta = Reg(env.pool.integers().pop());
        const Reg tb = Reg(env.pool.integers().pop());
        const Reg tmp = Reg(env.pool.integers().pop());
        const int M_times_N = M * N;

        combinator.nested_1d(stream::policy_t::zero(), a_iter->ref(),
                b_iter->ref(), [&](const index_t &a_idx, const index_t &b_idx) {
            const int k = a_idx[0];
            const int mm = a_idx[1];
            const int nn = b_idx[1];
            const int off = (k * M_times_N + mm * N + nn) * 8;
            load.xload(ta, a_iter->pointer(a_idx), 8);
            load.xload(tb, b_iter->pointer(b_idx), 8);
            gen.add(tmp, ta, tb);
            store.xstore(tmp, pointer_t(out_ptr, off), 8);
        });
        gen.ret();
    };
}

/// nested_1d() over two plain iterator streams computes every (k, m, n) sum
TEST(nest_bare_iterator, computes_every_sum_of_products) {
    const int K = 2, M = 3, N = 4;

    rv64_jit_probe_t kernel(nest_bare_iterator_probe(K, M, N));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int64_t> a(K * M), b(K * N), out(K * M * N, 0);
    for (int i = 0; i < K * M; ++i)
        a[i] = 100 + i;
    for (int i = 0; i < K * N; ++i)
        b[i] = 10000 + i;

    kernel(a.data(), b.data(), out.data());

    for (int k = 0; k < K; ++k)
        for (int mm = 0; mm < M; ++mm)
            for (int nn = 0; nn < N; ++nn) {
                const auto want = a[k * M + mm] + b[k * N + nn];
                const auto got = out[(k * M + mm) * N + nn];
                ASSERT_EQ(got, want) << "at k=" << k << " m=" << mm;
            }
}

} // namespace rvjit
