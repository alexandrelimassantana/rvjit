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

namespace rvjit {

using namespace Xbyak_riscv;

using branch_t = control_flow::branch_t;
using cond_t = control_flow::cond_t;
using distance_t = control_flow::distance_t;
using on_enter_t = control_flow::on_enter_t;
using on_index_t = control_flow::on_index_t;

namespace {

/// Emulate rvjit branch conditions
bool evaluate(cond_t cond, int64_t lhs, int64_t rhs) {
    switch (cond) {
        case cond_t::eq: return lhs == rhs;
        case cond_t::ne: return lhs != rhs;
        case cond_t::lt: return lhs < rhs;
        case cond_t::le: return lhs <= rhs;
        case cond_t::gt: return lhs > rhs;
        case cond_t::ge: return lhs >= rhs;
    }
    return false;
}

/// Emits `min_bytes` of filler instructions to create artificial distance
///
/// @note Used to test long branch expressions
void fill_nops(Xbyak_riscv::CodeGenerator &cg, size_t min_bytes) {
    const size_t start = cg.getSize();
    while (cg.getSize() - start < min_bytes)
        cg.addi(zero, zero, 0);
}

/// Builds an oversized probe from `body` and expects `create_kernel()` to
/// throw
void expect_out_of_reach(rv64_jit_probe_t::generate_fn_t body) {
    rv64_jit_probe_t kernel(1 << 15, std::move(body));
    EXPECT_THROW(kernel.create_kernel(), Error);
}

} // namespace

class control_flow_if_then_t
    : public ::testing::TestWithParam<
              std::tuple<cond_t, distance_t, int64_t, int64_t>> {};

/// Correctness expectations regarding the testing conditions of rvjit branches
TEST_P(control_flow_if_then_t, branch_taken) {
    cond_t cond;
    distance_t dist;
    int64_t lhs, rhs;
    std::tie(cond, dist, lhs, rhs) = GetParam();

    // Evaluate correctness of the `if` instruction emission
    rv64_jit_probe_t kernel([cond, dist](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg lhs = a0;
        const Reg rhs = a1;
        const Reg out = a2;
        const Reg res = a3;

        gen.mv(res, x0);
        env.cf.if_(branch_t(lhs, rhs, cond, dist), [&] { gen.li(res, 1); });
        gen.sd(res, out, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    run_timed(kernel, lhs, rhs, &out);

    const int64_t expect = evaluate(cond, lhs, rhs) ? 1 : 0;
    ASSERT_EQ(out, expect);
}

/// All branch types, distances, and logical relationships within arguments
INSTANTIATE_TEST_SUITE_P(conditions, control_flow_if_then_t,
        ::testing::Combine(::testing::Values(cond_t::eq, cond_t::ne, cond_t::lt,
                                   cond_t::le, cond_t::gt, cond_t::ge),
                ::testing::Values(distance_t::short_, distance_t::medium),
                ::testing::Values<int64_t>(-5, 0, 5),
                ::testing::Values<int64_t>(-5, 0, 5)));

class control_flow_if_else_t
    : public ::testing::TestWithParam<std::tuple<cond_t, int64_t, int64_t>> {};

/// Exactly one of the two `if_` branches runs, depending on the condition
TEST_P(control_flow_if_else_t, one_path_runs) {
    cond_t cond;
    int64_t lhs, rhs;
    std::tie(cond, lhs, rhs) = GetParam();

    // if then else: exactly one path runs
    rv64_jit_probe_t kernel([cond](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg lhs = a0;
        const Reg rhs = a1;
        const Reg out = a2;
        const Reg res = a3;

        env.cf.if_else(branch_t(lhs, rhs, cond), [&] { gen.li(res, 1); },
                [&] { gen.li(res, 2); });
        gen.sd(res, out, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    run_timed(kernel, lhs, rhs, &out);

    const int64_t expect = evaluate(cond, lhs, rhs) ? 1 : 2;
    ASSERT_EQ(out, expect);
}

/// All branch types and logical relationships within arguments
INSTANTIATE_TEST_SUITE_P(conditions, control_flow_if_else_t,
        ::testing::Combine(::testing::Values(cond_t::eq, cond_t::ne, cond_t::lt,
                                   cond_t::le, cond_t::gt, cond_t::ge),
                ::testing::Values<int64_t>(-5, 0, 5),
                ::testing::Values<int64_t>(-5, 0, 5)));

class control_flow_while_t
    : public ::testing::TestWithParam<std::tuple<int64_t, int>> {};

/// `while_` runs its body ceil(n/step) times, advancing the iterator by step
TEST_P(control_flow_while_t, counts_and_final_iter) {
    int64_t n;
    int step;
    std::tie(n, step) = GetParam();

    // while less than step: counts how many times the body runs
    rv64_jit_probe_t kernel([step](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg N = a0;
        const Reg out_count = a1;
        const Reg out_iters = a2;
        const Reg iters = a3;
        const Reg n = a4;

        gen.mv(iters, x0);
        gen.mv(n, x0);
        env.cf.while_(branch_t(iters, N, cond_t::lt), const_t(step),
                [&] { gen.addi(n, n, 1); });
        gen.sd(n, out_count, 0);
        gen.sd(iters, out_iters, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t count = -1;
    int64_t iters = -1;
    run_timed(kernel, n, &count, &iters);

    const int64_t expect_count = (n + step - 1) / step;
    ASSERT_EQ(count, expect_count);
    ASSERT_EQ(iters, expect_count * step);
}

/// Trip counts spanning zero, non-multiples, and exact multiples of `step`
INSTANTIATE_TEST_SUITE_P(sizes, control_flow_while_t,
        ::testing::Combine(::testing::Values<int64_t>(0, 1, 5, 7, 10, 16),
                ::testing::Values(1, 2, 4)));

class control_flow_switch_case_t
    : public ::testing::TestWithParam<std::tuple<int, int64_t>> {};

/// `switch_case` runs exactly the case body matching `id`, and no other
TEST_P(control_flow_switch_case_t, matching_case_runs) {
    int n_cases;
    int64_t id;
    std::tie(n_cases, id) = GetParam();

    // switch_case: exactly the matching case executes
    rv64_jit_probe_t kernel([n_cases](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg id = a0;
        const Reg out = a1;
        const Reg tmp = a2;
        const Reg result = a4;

        gen.mv(result, x0);
        env.cf.switch_case(n_cases, id, tmp, [&](int i) { gen.li(result, i); });
        gen.sd(result, out, 0);
        gen.ret();
    });

    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = -1;
    run_timed(kernel, id, &out);

    ASSERT_EQ(out, id);
}

/// Case counts spanning the direct, if/else, and jump-table dispatch paths
INSTANTIATE_TEST_SUITE_P(cases, control_flow_switch_case_t,
        ::testing::ValuesIn(std::vector<std::tuple<int, int64_t>> {
                {1, 1},
                {2, 1},
                {2, 2},
                {3, 1},
                {3, 2},
                {3, 3},
                {16, 1},
                {16, 8},
                {16, 16},
                {32, 1},
                {32, 16},
                {32, 32},
        }));

/// if: a `short_` branch whose body overflows the addressing range
TEST(control_flow_branch_distance_t, if_out_of_reach) {
    expect_out_of_reach([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        env.cf.if_(branch_t(a0, a1, cond_t::eq, distance_t::short_),
                [&] { fill_nops(gen, 1 << 12); });
        gen.ret();
    });
}

/// if then else: a `short_` branch whose body overflows the addressing range
TEST(control_flow_branch_distance_t, if_then_else_out_of_reach) {
    expect_out_of_reach([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        env.cf.if_else(branch_t(a0, a1, cond_t::eq, distance_t::short_),
                [&] { fill_nops(gen, 1 << 12); }, [&] {});
        gen.ret();
    });
}

/// while: a `short_` branch whose body overflows the addressing range
TEST(control_flow_branch_distance_t, while_out_of_reach) {
    expect_out_of_reach([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg iters = a2;
        gen.mv(iters, x0);
        env.cf.while_(branch_t(iters, a0, cond_t::lt, distance_t::short_),
                const_t(1), [&] { fill_nops(gen, 1 << 12); });
        gen.ret();
    });
}

} // namespace rvjit
