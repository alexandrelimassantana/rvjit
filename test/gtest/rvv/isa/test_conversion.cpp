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

#include <string>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/arithmetic.hpp"
#include "rvjit/interface/isa/conversion.hpp"

namespace rvjit {

namespace {

/// The planned steps of each pair, and the pairs no chain converts
TEST(conversion_plan, steps) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    const auto vec = form_t::vector;
    const auto kinds = [&](type_t from, type_t to) {
        conversion_t c;
        std::string s;
        if (!c.plan(from, to, vec, ar)) return std::string("-");
        for (int i = 0; i < c.size(); ++i) {
            const conversion::kind_t k = c.step(i).kind;
            s += k == conversion::kind_t::extend       ? 'E'
                    : k == conversion::kind_t::widen   ? 'W'
                    : k == conversion::kind_t::convert ? 'C'
                                                       : 'N';
        }
        return s;
    };
    EXPECT_EQ(kinds(type_t::f32, type_t::f32), "");
    EXPECT_EQ(kinds(type_t::f16, type_t::f32), "W");
    EXPECT_EQ(kinds(type_t::f32, type_t::bf16), "N");
    EXPECT_EQ(kinds(type_t::s16, type_t::s32), "W");
    EXPECT_EQ(kinds(type_t::s8, type_t::s32), "E");
    EXPECT_EQ(kinds(type_t::s32, type_t::f32), "C");
    EXPECT_EQ(kinds(type_t::u8, type_t::f32), "EC");
    EXPECT_EQ(kinds(type_t::f32, type_t::s8), "CNN");
    EXPECT_EQ(kinds(type_t::f16, type_t::s8), "CN");
    EXPECT_EQ(kinds(type_t::s32, type_t::s8), "NN");
    EXPECT_EQ(kinds(type_t::f16, type_t::f64), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::u8), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::u32), "-");
    EXPECT_EQ(kinds(type_t::u8, type_t::s8), "-");
    EXPECT_EQ(kinds(type_t::s32, type_t::f16), "-");
    EXPECT_EQ(kinds(type_t::bf16, type_t::s16), "-");
}

/// Each step names the width its configuration runs at
TEST(conversion_plan, steps_configure_at_their_widths) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    conversion_t c;
    ASSERT_TRUE(c.plan(type_t::f16, type_t::f32, form_t::vector, ar));
    EXPECT_EQ(c.step(0).config(), type_t::f16);
    ASSERT_TRUE(c.plan(type_t::u8, type_t::f32, form_t::vector, ar));
    EXPECT_EQ(c.step(0).config(), type_t::u32);
    EXPECT_EQ(c.step(1).config(), type_t::f32);
    ASSERT_TRUE(c.plan(type_t::f32, type_t::s8, form_t::vector, ar));
    EXPECT_EQ(c.step(0).config(), type_t::s32);
    EXPECT_EQ(c.step(2).config(), type_t::s8);
    EXPECT_EQ(c.ratio(), 4);
}

/// A failed plan leaves the conversion empty
TEST(conversion_plan, a_failed_plan_is_empty) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    conversion_t c;
    ASSERT_TRUE(c.plan(type_t::f16, type_t::f32, form_t::vector, ar));
    EXPECT_FALSE(c.plan(type_t::s32, type_t::u32, form_t::vector, ar));
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(c.from(), type_t::none);
}

/// A load plans from the memory type, a store into it
TEST(conversion_plan, loads_and_stores_follow_the_value_types) {
    Xbyak_riscv::CodeGenerator gen;
    env_t env(gen);
    rvv::arithmetic::arithmetic_t ar(env.e);
    const type_parameter_t t(type_t::f16, type_t::f32);
    conversion_t c;
    ASSERT_TRUE(c.plan_load(t, form_t::vector, ar));
    EXPECT_EQ(c.from(), type_t::f16);
    EXPECT_EQ(c.to(), type_t::f32);
    ASSERT_TRUE(c.plan_store(t, form_t::vector, ar));
    EXPECT_EQ(c.from(), type_t::f32);
    EXPECT_EQ(c.to(), type_t::f16);
}

} // namespace
} // namespace rvjit
