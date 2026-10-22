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
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/induction/induction.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

namespace {

using gen_t = rv64_jit_test_generator_t;

const Reg var = t2, slot = t3, opr = t4, tmp = t5, other = t6, other_slot = a2;

/// The values registers held before and after an induction
struct result_t {
    std::vector<std::pair<Reg, int64_t>> in;
    std::vector<int64_t> out;

    /// The value `r` ended with
    int64_t operator[](const Reg &r) const {
        for (size_t i = 0; i < in.size(); ++i)
            if (reg_name(in[i].first) == reg_name(r)) return out[i];
        return 0;
    }
};

/// Runs `binds` around a body that clobbers registers, returning their values
///
/// @param in The registers loaded beforehand, with their values
/// @param binds The bindings driven around the body
/// @param clobber The registers the body sets to -1
/// @param stack Whether the kernel reserves stack space for memory slots
result_t run(const std::vector<std::pair<Reg, int64_t>> &in,
        const std::vector<bind_t> &binds, const std::vector<Reg> &clobber,
        bool stack = false) {
    rv64_jit_probe_t kernel([=](gen_t &gen) {
        env_t env(gen);
        for (size_t i = 0; i < in.size(); ++i)
            gen.ld(in[i].first, a0, static_cast<int>(8 * i));

        if (stack) gen.addi(sp, sp, -16);
        induction_t ind;
        for (const bind_t &b : binds)
            ind.push_back(b);
        ind(env.xfer, [&] {
            for (const Reg &r : clobber)
                gen.li(r, -1);
        });
        if (stack) gen.addi(sp, sp, 16);

        for (size_t i = 0; i < in.size(); ++i)
            gen.sd(in[i].first, a0, static_cast<int>(8 * i));
        gen.ret();
    });
    EXPECT_TRUE(kernel.create_kernel());

    result_t r;
    r.in = in;
    for (const auto &v : in)
        r.out.push_back(v.second);
    kernel(r.out.data());
    return r;
}

bind_t slotted(const Reg &reg, const Reg &to, const step_t &step = {}) {
    bind_t b;
    b.reg = reg;
    b.slot = slot_t::buffer(to);
    b.step = step;
    return b;
}

} // namespace

/// A value the body clobbers comes back from its slot
TEST(induction_values, value_is_restored) {
    const auto r = run({{var, 7}}, {slotted(var, slot)}, {var});
    EXPECT_EQ(r[var], 7);
}

/// A zero step still brings the value back
TEST(induction_values, zero_step_restores) {
    const auto r = run(
            {{var, 7}}, {slotted(var, slot, step_t::plain(const_t(0)))}, {var});
    EXPECT_EQ(r[var], 7);
}

/// Steps add or subtract an immediate or a register
TEST(induction_values, add_and_sub_steps) {
    const std::vector<std::pair<Reg, int64_t>> in = {{var, 100}, {opr, 5}};
    {
        const bind_t b = slotted(var, slot, step_t::plain(const_t(8)));
        const auto r = run(in, {b}, {var});
        EXPECT_EQ(r[var], 108);
    }
    {
        const bind_t b = slotted(var, slot, step_t::plain(const_t(opr)));
        const auto r = run(in, {b}, {var});
        EXPECT_EQ(r[var], 105);
    }
    {
        const bind_t b = slotted(var, slot, step_t::sub(const_t(opr)));
        const auto r = run(in, {b}, {var});
        EXPECT_EQ(r[var], 95);
    }
}

/// A macc step adds the product of its factors
TEST(induction_values, macc_register_operand) {
    const auto r = run({{var, 100}, {opr, 5}},
            {slotted(var, slot, step_t::macc(const_t(opr), const_t(8), tmp))},
            {var});
    EXPECT_EQ(r[var], 100 + 5 * 8);
}

/// A product that overflows an immediate is still added in full
TEST(induction_values, macc_overflowing_product) {
    const auto r = run({{var, 10000}},
            {slotted(var, slot, step_t::macc(const_t(-1), const_t(4096), tmp))},
            {var});
    EXPECT_EQ(r[var], 10000 - 4096);
}

/// A macc step with no scratch register of its own works as well
TEST(induction_values, macc_without_scratch) {
    const auto r = run({{var, 100}, {opr, 5}},
            {slotted(var, slot, step_t::macc(const_t(opr), const_t(8)))},
            {var});
    EXPECT_EQ(r[var], 100 + 5 * 8);
}

/// A value saved to the stack comes back advanced
TEST(induction_values, memory_slot) {
    bind_t b = slotted(var, slot, step_t::plain(const_t(8)));
    b.slot = slot_t::memory(pointer_t(sp, 0));
    const auto r = run({{var, 100}}, {b}, {var}, true);
    EXPECT_EQ(r[var], 108);
}

/// A value reloaded from a read-only origin is advanced from it
TEST(induction_values, copy_origin) {
    bind_t b;
    b.reg = var;
    b.origin = source_t::copy(slot);
    b.step = step_t::plain(const_t(8));
    const auto r = run({{var, 0}, {slot, 50}}, {b}, {var});
    EXPECT_EQ(r[var], 58);
    EXPECT_EQ(r[slot], 50);
}

/// A step reading its own register adds the restored value
TEST(induction_values, step_reading_its_own_register) {
    const auto r = run({{var, 7}},
            {slotted(var, slot, step_t::plain(const_t(var)))}, {var});
    EXPECT_EQ(r[var], 14);
}

/// A step reading an unstepped register sees its restored value
TEST(induction_values, step_reading_an_unstepped_register) {
    const auto r = run({{var, 100}, {other, 1}},
            {slotted(var, slot, step_t::plain(const_t(other))),
                    slotted(other, other_slot)},
            {var, other});
    EXPECT_EQ(r[var], 101);
    EXPECT_EQ(r[other], 1);
}

/// A step reading the register of an earlier binding sees its updated value
TEST(induction_values, step_reading_an_earlier_register) {
    const auto r = run({{var, 100}, {other, 1}},
            {slotted(var, slot, step_t::plain(const_t(8))),
                    slotted(other, other_slot, step_t::plain(const_t(var)))},
            {var, other});
    EXPECT_EQ(r[var], 108);
    EXPECT_EQ(r[other], 109);
}

/// A step reading the register of a later binding sees its restored value
TEST(induction_values, step_reading_a_later_register) {
    const auto r = run({{var, 100}, {other, 1}},
            {slotted(other, other_slot, step_t::plain(const_t(var))),
                    slotted(var, slot, step_t::plain(const_t(8)))},
            {var, other});
    EXPECT_EQ(r[other], 101);
    EXPECT_EQ(r[var], 108);
}

/// An inactive binding leaves the value as the body left it
TEST(induction_values, inactive_binding_is_skipped) {
    bind_t b = slotted(var, slot, step_t::plain(const_t(8)));
    b.active = false;
    const auto r = run({{var, 100}}, {b}, {var});
    EXPECT_EQ(r[var], -1);
}

} // namespace rvjit
