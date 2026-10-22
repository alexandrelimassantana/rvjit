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

#include "gtest/gtest.h"

#include "rvjit/interface/linalg/matmul/postop/builtin.hpp"
#include "rvjit/interface/linalg/matmul/postop/chain.hpp"

namespace rvjit {

using linalg::matmul::postop::scalar_bits;
using linalg::matmul::postop::scalar_mode_t;
using linalg::matmul::postop::scalar_t;

// scalar_bits(): the bit patterns literal alpha/beta values materialize as

/// f32 bits are the IEEE pattern, sign-extended into the immediate
TEST(postop_scalar_bits, f32) {
    int64_t bits = 0;
    ASSERT_TRUE(scalar_bits(1.0, type_t::f32, bits));
    EXPECT_EQ(bits, 0x3f800000);
    ASSERT_TRUE(scalar_bits(-2.0, type_t::f32, bits));
    EXPECT_EQ(static_cast<uint32_t>(bits), 0xc0000000u);
}

/// f16 rounds to nearest even and saturates to infinity
TEST(postop_scalar_bits, f16) {
    int64_t bits = 0;
    ASSERT_TRUE(scalar_bits(1.0, type_t::f16, bits));
    EXPECT_EQ(bits, 0x3c00);
    ASSERT_TRUE(scalar_bits(-3.0, type_t::f16, bits));
    EXPECT_EQ(bits, 0xc200);
    ASSERT_TRUE(scalar_bits(1.0009765625, type_t::f16, bits)); // 1 + 2^-10
    EXPECT_EQ(bits, 0x3c01);
    ASSERT_TRUE(scalar_bits(1.00048828125, type_t::f16, bits)); // a tie: even
    EXPECT_EQ(bits, 0x3c00);
    ASSERT_TRUE(scalar_bits(1e9, type_t::f16, bits));
    EXPECT_EQ(bits, 0x7c00);
}

/// f64 only fits an immediate when its upper half is zero
TEST(postop_scalar_bits, f64) {
    int64_t bits = 0;
    EXPECT_TRUE(scalar_bits(0.0, type_t::f64, bits));
    EXPECT_FALSE(scalar_bits(1.0, type_t::f64, bits));
}

/// Integers must fit the 32-bit immediate
TEST(postop_scalar_bits, ints) {
    int64_t bits = 0;
    ASSERT_TRUE(scalar_bits(-3.0, type_t::s16, bits));
    EXPECT_EQ(bits, -3);
    EXPECT_FALSE(scalar_bits(4.0e9, type_t::s64, bits));
    EXPECT_FALSE(scalar_bits(1.0, type_t::none, bits));
}

/// Literal 0 and 1 fold; anything else, and a runtime source, is a value
TEST(postop_scalar, classify) {
    {
        const auto mode = scalar_t::classify(scalar_parameter_t(0.0));
        EXPECT_TRUE(mode == scalar_mode_t::zero);
    }
    {
        const auto mode = scalar_t::classify(scalar_parameter_t(1.0));
        EXPECT_TRUE(mode == scalar_mode_t::one);
    }
    {
        const auto mode = scalar_t::classify(scalar_parameter_t(2.5));
        EXPECT_TRUE(mode == scalar_mode_t::value);
    }
    {
        const auto mode
                = scalar_t::classify(scalar_parameter_t(source_t(reg_t())));
        EXPECT_TRUE(mode == scalar_mode_t::value);
    }
}

/// A float scalar needs a float register on top of the bits' integer one
TEST(postop_scalar, needs) {
    EXPECT_EQ(scalar_t::needs(type_t::f32).floats, 1);
    EXPECT_EQ(scalar_t::needs(type_t::s32).floats, 0);
    EXPECT_EQ(scalar_t::needs(type_t::s32).ints, 1);
}

/// merge() raises each need to the larger of the two
TEST(postop_needs, merge_takes_the_maximum) {
    linalg::matmul::needs_t a, b;
    a.accs = 2;
    b.accs = 1;
    b.ints = 3;
    b.m_offset = true;
    b.accs_wanted = 5;
    b.mask = true;
    a.merge(b);
    EXPECT_EQ(a.accs, 2);
    EXPECT_EQ(a.accs_wanted, 5);
    EXPECT_TRUE(a.mask);
    EXPECT_EQ(a.ints, 3);
    EXPECT_TRUE(a.m_offset);
    EXPECT_FALSE(a.args);
}

namespace {
linalg::matmul::plan_t chain_plan(int flags) {
    linalg::matmul::plan_t p;
    p.flags = flags;
    p.alpha = scalar_parameter_t(2.5);
    p.beta = scalar_parameter_t(source_t(reg_t(10)));
    p.bias = parameter_t(source_t(reg_t(11)));
    return p;
}
} // namespace

/// The needs of a plan's chain are known without building a kernel
TEST(postop_chain, needs_follow_the_active_entries) {
    using namespace linalg::matmul;
    needs_t n;

    ASSERT_TRUE(chain_needs(chain_plan(0), 2, 3, n));
    EXPECT_EQ(n.accs, 0);

    // The bias holds one vector per M vector of the tile, and the M offset
    ASSERT_TRUE(chain_needs(chain_plan(with_bias), 3, 1, n));
    EXPECT_EQ(n.accs, 3);
    EXPECT_TRUE(n.m_offset);

    // A runtime beta loads C beside the accumulators, one register at least,
    // two when free to overlap the loads, and needs its value
    ASSERT_TRUE(chain_needs(chain_plan(with_beta), 3, 1, n));
    EXPECT_EQ(n.accs, 1);
    EXPECT_EQ(n.accs_wanted, 2);
    EXPECT_EQ(n.floats, 1);

    // The entries share their registers: the largest need wins
    ASSERT_TRUE(chain_needs(chain_plan(with_beta | with_bias), 4, 1, n));
    EXPECT_EQ(n.accs, 4);
}

/// A literal flag word leaves out the entries it disables
TEST(postop_chain, a_literal_flag_word_disables_entries) {
    using namespace linalg::matmul;
    plan_t p = chain_plan(with_beta | with_bias);
    p.rt_flags = parameter_t(with_bias);
    needs_t n;
    ASSERT_TRUE(chain_needs(p, 3, 1, n));
    EXPECT_EQ(n.accs, 3);
    EXPECT_EQ(n.floats, 0);
}

/// A feature without what it needs is an inconsistent plan
TEST(postop_chain, inconsistent_plans_are_rejected) {
    using namespace linalg::matmul;
    plan_t p;
    p.flags = with_bias;
    needs_t n;
    EXPECT_FALSE(chain_needs(p, 1, 1, n));
}

} // namespace rvjit
