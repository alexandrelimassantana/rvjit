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

#include "interface/induction/transfer_mock.hpp"
#include "isa_any_registers.hpp"
#include "rvjit/interface/induction/induction.hpp"

namespace rvjit {

using namespace isa_any;

namespace {

const reg_t var = r7, slot = r28, other = r29, other_slot = r30, operand = r31;

/// Runs `binds` around a body that records a `body` row, returning the rows
///
/// @details Rows are `verb,mode,dest,src1,src2`, registers shown by label:
///     - verbs: `put` (save), `get` (reload), `advance` (update), `body`
///     - put modes, by slot: `reg`, `mem`
///     - get modes, by source: `reg`, `imm`, `mem`
///     - advance modes, by step: `imm`, `reg`, `sub_imm`, `sub_reg`, `macc`
///     - fields: `dest` is written, `src1` read, `src2` the step operand
///     - operands: a label, an immediate, `base+offset`, or `a*b` for macc
///
/// @note The rows follow a leading newline, so a table starts on its own line
std::string run(std::initializer_list<bind_t> binds) {
    recording_transfer_t xfer;
    xfer.name(var, "var");
    xfer.name(slot, "slot");
    xfer.name(other, "other");
    xfer.name(other_slot, "other_slot");
    xfer.name(operand, "operand");
    xfer.name(stack, "stack");
    induction_t ind(binds);
    ind(xfer, [&] { xfer.mark("body"); });
    return "\n" + xfer.csv();
}

bind_t slotted(const reg_t &reg, const reg_t &to, const step_t &step = {}) {
    bind_t b;
    b.reg = reg;
    b.slot = slot_t::buffer(to);
    b.step = step;
    return b;
}

} // namespace

/// A stepped value saved in a register is reloaded and advanced in one call
TEST(induction_fusion, stepped_buffer_slot_fuses) {
    const std::string want = R"(
put,reg,slot,var
body
advance,imm,var,slot,8
)";
    EXPECT_EQ(run({slotted(var, slot, step_t::plain(const_t(8)))}), want);
}

/// A read-only origin needs no save, only the fused reload
TEST(induction_fusion, copy_origin_fuses) {
    bind_t b;
    b.reg = var;
    b.origin = source_t::copy(slot);
    b.step = step_t::plain(const_t(8));
    {
        const std::string want = R"(
body
advance,imm,var,slot,8
)";
        EXPECT_EQ(run({b}), want);
    }
}

/// A step may read registers that no binding holds
TEST(induction_fusion, step_reading_an_unbound_register_fuses) {
    const step_t step = step_t::macc(const_t(operand), const_t(8));
    {
        const std::string want = R"(
put,reg,slot,var
body
advance,macc,var,slot,operand*8
)";
        EXPECT_EQ(run({slotted(var, slot, step)}), want);
    }
}

/// A memory slot is loaded back into the value before it is advanced
TEST(induction_fusion, memory_slot_is_not_fused) {
    bind_t b = slotted(var, slot, step_t::plain(const_t(8)));
    b.slot = slot_t::memory(pointer_t(stack, 8));
    {
        const std::string want = R"(
put,mem,stack+8,var
body
get,mem,var,stack+8
advance,imm,var,var,8
)";
        EXPECT_EQ(run({b}), want);
    }
}

/// A step may read a bound register that has no update of its own
TEST(induction_fusion, step_reading_an_unstepped_register_fuses) {
    const bind_t reader = slotted(var, slot, step_t::plain(const_t(other)));
    const bind_t bound = slotted(other, other_slot);
    {
        const std::string want = R"(
put,reg,slot,var
put,reg,other_slot,other
body
get,reg,other,other_slot
advance,reg,var,slot,other
)";
        EXPECT_EQ(run({reader, bound}), want);
    }
}

/// A step reading the register of an earlier binding sees its updated value
TEST(induction_fusion, step_reading_an_earlier_register_sees_its_update) {
    const bind_t first = slotted(var, slot, step_t::plain(const_t(8)));
    const bind_t reader
            = slotted(other, other_slot, step_t::plain(const_t(var)));
    {
        const std::string want = R"(
put,reg,slot,var
put,reg,other_slot,other
body
advance,imm,var,slot,8
advance,reg,other,other_slot,var
)";
        EXPECT_EQ(run({first, reader}), want);
    }
}

/// A step reading a later binding's register sees its restored value
TEST(induction_fusion, step_reading_a_later_register_sees_its_restore) {
    const bind_t reader
            = slotted(other, other_slot, step_t::plain(const_t(var)));
    const bind_t later = slotted(var, slot, step_t::plain(const_t(8)));
    {
        const std::string want = R"(
put,reg,other_slot,other
put,reg,slot,var
body
get,reg,var,slot
advance,reg,other,other_slot,var
advance,imm,var,var,8
)";
        EXPECT_EQ(run({reader, later}), want);
    }
}

/// A step reading its own register is not fused
TEST(induction_fusion, step_reading_its_own_register_is_not_fused) {
    const std::string want = R"(
put,reg,slot,var
body
get,reg,var,slot
advance,reg,var,var,var
)";
    EXPECT_EQ(run({slotted(var, slot, step_t::plain(const_t(var)))}), want);
}

/// Independent bindings are saved together, then reloaded together
TEST(induction_fusion, independent_bindings_run_in_passes) {
    const bind_t first = slotted(var, slot, step_t::plain(const_t(8)));
    const bind_t second = slotted(other, other_slot, step_t::plain(const_t(4)));
    {
        const std::string want = R"(
put,reg,slot,var
put,reg,other_slot,other
body
advance,imm,var,slot,8
advance,imm,other,other_slot,4
)";
        EXPECT_EQ(run({first, second}), want);
    }
}

/// A value without a step is only reloaded
TEST(induction_fusion, stepless_binding_only_restores) {
    const std::string want = R"(
put,reg,slot,var
body
get,reg,var,slot
)";
    EXPECT_EQ(run({slotted(var, slot)}), want);
}

/// An inactive binding takes no part
TEST(induction_fusion, inactive_binding_is_skipped) {
    bind_t b = slotted(var, slot, step_t::plain(const_t(8)));
    b.active = false;
    {
        const std::string want = R"(
body
)";
        EXPECT_EQ(run({b}), want);
    }
}

} // namespace rvjit
