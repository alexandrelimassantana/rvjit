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
#include <set>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "stream_mock.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;
using rv64_iterator_t = rvjit::rv64::stream::iterator::iterator_t;

using stream::on_interleave_t;
using stream::on_nest_t;
using stream::policy_t;
using stream::ravel;
using stream::scheduler_t;
using stream::unravel;

// JIT-executed: nested() driving real memory-operand codegen

namespace {

/// Builds a probe that drives nested() over two real memory operands (A: a
/// {K, M} row-major matrix, B: a {K, N} row-major matrix), storing
/// `A[k,m] + B[k,n]` at every (k, m, n)
rv64_jit_probe_t::generate_fn_t nest_jit_probe(int K, int M, int N, int delay) {
    return [K, M, N, delay](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        common::stream::combinator_t combinator;
        rvv::memory::store_t store(env.e);

        const Reg a_ptr = a0, b_ptr = a1, out_ptr = a2;

        // A: {K, M}, row-major (stride K = M*8, stride M = 8)
        space_t a_space;
        a_space.axis(K, const_t(M * 8)).axis(M, const_t(8));
        // Non-outermost direct axes (here, M) always cost 1 register
        // regardless of end-state, so ntmp=0 would leave this axis
        // unresolved
        const auto ntmp = env.pool.integers().available();

        // B: {K, N}, row-major (stride K = N*8, stride N = 8)
        space_t b_space;
        b_space.axis(K, const_t(N * 8)).axis(N, const_t(8));
        memory::op_t op = memory::op_t().scalar().load().as_type(type_t::s64);

        // A: {K, 1} -- mirrors rvv::linalg::matmul::matmul_t's own A shape (M folded
        // into vector lanes, not a real nest axis here either); a deferred
        // cb call may reach back across a k boundary, so its K axis ring
        // is 2 deep, tolerating a delay strictly less than one whole row
        // (M*N) crossing into the previous k
        namespace mo = rvjit::stream::memory_operand;
        mo::spec_t a_spec;
        a_spec.op = op;
        a_spec.origin = pointer_t(a_ptr, 0);
        a_spec.space = a_space;
        a_spec.update = update_t::advance(0);
        mo::variant_t a_variant;
        a_variant.strategy = rv64::stream::iterator::guided;
        a_variant.ntmp = ntmp;
        a_variant.ring = buffer_map_t({2, 1});
        auto *a_stream
                = env.pm.streams().memory_operand().create(a_spec, a_variant);
        ASSERT_TRUE(a_stream->is_valid());

        // B: reused across every m within a k, so its ring must hold all N
        // positions; 2 deep on the shared K axis for the same cross-row
        // reason as A's own ring above
        mo::spec_t b_spec;
        b_spec.op = op;
        b_spec.origin = pointer_t(b_ptr, 0);
        b_spec.space = b_space;
        b_spec.update = update_t::advance(0);
        mo::variant_t b_variant;
        b_variant.strategy = rv64::stream::iterator::guided;
        b_variant.ntmp = ntmp;
        b_variant.ring = buffer_map_t({2, N});
        auto *b_stream
                = env.pm.streams().memory_operand().create(b_spec, b_variant);
        ASSERT_TRUE(b_stream->is_valid());

        const Reg tmp = Reg(env.pool.integers().pop());
        const int M_times_N = M * N;

        combinator.nested_1d(policy_t::literal(delay), a_stream->ref(),
                b_stream->ref(),
                [&](const index_t &a_idx, const index_t &b_idx) {
            const int k = a_idx[0];
            const int mm = a_idx[1];
            const int nn = b_idx[1];
            const int off = (k * M_times_N + mm * N + nn) * 8;
            gen.add(tmp, Reg(a_stream->get(a_idx)), Reg(b_stream->get(b_idx)));
            store.xstore(tmp, pointer_t(out_ptr, off), 8);
        });
        gen.ret();
    };
}

} // namespace

class stream_nest_jit_t : public ::testing::TestWithParam<int> {};

/// nested() with real memory streams computes every (k, m, n) sum, any delay
TEST_P(stream_nest_jit_t, computes_every_sum_of_products) {
    const int K = 2, M = 1, N = 4;
    const int delay = GetParam();

    rv64_jit_probe_t kernel(1 << 16, nest_jit_probe(K, M, N, delay));
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
                const int64_t want = a[k * M + mm] + b[k * N + nn];
                const auto got = out[k * M * N + mm * N + nn];
                ASSERT_EQ(got, want) << "at " << k << "," << mm << "," << nn;
            }
}

/// Delays 0, 1 and 3, the largest a ring depth of 2 tolerates, past a K-row
INSTANTIATE_TEST_SUITE_P(delays, stream_nest_jit_t, ::testing::Values(0, 2, 3));

} // namespace rvjit
