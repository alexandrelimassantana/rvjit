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
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/variable.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

// transfer_t: binding-driven stack-slot sharing

/// A `bind_t` on a stack slot round-trips a clobbered register
TEST(persistence, binding_round_trips_through_stack_slot) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg a = Reg(env.pool.integers().pop());
        gen.li(a, 66);

        rvjit::bind_t entry;
        entry.reg = a;
        entry.slot = env.stack.new_slot();
        env.stack.reserve_stack();
        env.xfer.preserve(entry);
        gen.li(a, -1); // clobber a -- must be recovered from the shared slot
        env.xfer.restore(entry);

        gen.sd(a, out_ptr, 0);
        env.stack.release_stack();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 66);
}

/// Nested stack reservations get disjoint slots, or a round trip would clash
TEST(persistence, reserve_stack_nests_stack_slots) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg ra = Reg(env.pool.integers().pop());
        const Reg rb = Reg(env.pool.integers().pop());
        gen.li(ra, 111);
        gen.li(rb, 222);

        rvjit::bind_t outer;
        outer.reg = ra;
        outer.slot = env.stack.new_slot();
        env.stack.reserve_stack();
        env.xfer.preserve(outer);

        rvjit::bind_t inner;
        inner.reg = rb;
        inner.slot = env.stack.new_slot();
        env.stack.reserve_stack();
        env.xfer.preserve(inner);

        gen.li(ra, -1);
        gen.li(rb, -1);

        env.xfer.restore(inner);
        env.stack.release_stack();
        env.xfer.restore(outer);
        env.stack.release_stack();

        gen.add(ra, ra, rb);
        gen.sd(ra, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 111 + 222);
}

/// Nested reserve_stack() pairs emit their own addi pair, like a flat total
TEST(persistence, reserve_stack_nested_emits_two_addi_pairs) {
    rv64_jit_probe_t nested([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        env.stack.new_stack_slot();
        env.stack.reserve_stack();
        env.stack.new_stack_slot();
        env.stack.reserve_stack();
        env.stack.release_stack();
        env.stack.release_stack();
        gen.ret();
    });
    rv64_jit_probe_t flat([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        env.stack.new_stack_slot();
        env.stack.new_stack_slot();
        env.stack.reserve_stack();
        env.stack.release_stack();
        gen.ret();
    });
    ASSERT_TRUE(nested.create_kernel());
    ASSERT_TRUE(flat.create_kernel());
    EXPECT_GT(nested.getSize(), flat.getSize());
}

// stack_t::new_stack_slot

/// new_stack_slot() bump-allocates 8 bytes per call and emits nothing
TEST(persistence, new_stack_slot_offset_sequence) {
    pointer_t offsets[3];
    rv64_jit_probe_t kernel([&offsets](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        offsets[0] = env.stack.new_stack_slot();
        offsets[1] = env.stack.new_stack_slot();
        offsets[2] = env.stack.new_stack_slot();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_EQ(offsets[0].base, Xbyak_riscv::sp);
    EXPECT_EQ(offsets[0].offset, 0);
    EXPECT_EQ(offsets[1].base, Xbyak_riscv::sp);
    EXPECT_EQ(offsets[1].offset, 8);
    EXPECT_EQ(offsets[2].base, Xbyak_riscv::sp);
    EXPECT_EQ(offsets[2].offset, 16);
}

// stack_t::reserve_stack / release_stack

/// reserve_stack() emits nothing when nothing was allocated, an addi otherwise
TEST(persistence, reserve_stack_emit_check) {
    rv64_jit_probe_t noop([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        env.stack.reserve_stack();
        env.stack.release_stack();
        gen.ret();
    });
    rv64_jit_probe_t allocated([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        env.stack.new_stack_slot();
        env.stack.reserve_stack();
        env.stack.release_stack();
        gen.ret();
    });
    ASSERT_TRUE(noop.create_kernel());
    ASSERT_TRUE(allocated.create_kernel());
    EXPECT_EQ(noop.getSize(), sizeof(int32_t));
    EXPECT_GT(allocated.getSize(), noop.getSize());
}

// stack_t::preserve()/restore(): emits the right sd/ld count
// and offsets

namespace {

// Emits only a source-excluded pool allocation + preserve() + restore() +
// ret(), so the sd/ld pairs are unambiguous
rv64_jit_probe_t::generate_fn_t preserve_restore_probe(
        std::vector<Reg> excl, int n_alloc) {
    return [excl, n_alloc](rv64_jit_test_generator_t &gen) {
        env_t env(gen,
                excluding_ints(buffer::buffer_t(excl.begin(), excl.end())));
        env.pool.integers().pop(n_alloc);
        env.stack.preserve();
        env.stack.restore();
        gen.ret();
    };
}

// RV64 S-type (SD) with rs1 == sp (x2): bits[6:0]=0x23, bits[14:12]=3
static bool is_sd_sp(uint32_t instr) {
    return (instr & 0x707fu) == 0x3023u && ((instr >> 15) & 0x1fu) == 2u;
}

// RV64 I-type (LD) with rs1 == sp (x2): bits[6:0]=0x03, bits[14:12]=3
static bool is_ld_sp(uint32_t instr) {
    return (instr & 0x707fu) == 0x3003u && ((instr >> 15) & 0x1fu) == 2u;
}

// S-type immediate: imm[4:0] = bits[11:7], imm[11:5] = bits[31:25]
static int sd_imm(uint32_t instr) {
    return static_cast<int>(((instr >> 7) & 0x1fu) | ((instr >> 25) << 5));
}

// S-type rs2 (data register): bits[24:20]
static int sd_rs2(uint32_t instr) {
    return static_cast<int>((instr >> 20) & 0x1fu);
}

// I-type immediate: bits[31:20]
static int ld_imm(uint32_t instr) {
    return static_cast<int>((instr >> 20) & 0xfffu);
}

// I-type rd (destination register): bits[11:7]
static int ld_rd(uint32_t instr) {
    return static_cast<int>((instr >> 7) & 0x1fu);
}

struct sp_mem_ops_t {
    std::vector<std::pair<int, int>> stores; // (reg_idx, stack_offset)
    std::vector<std::pair<int, int>> loads;
};

// Scans `k`'s emitted code for sd/ld instructions targeting sp, in
// encounter order
sp_mem_ops_t collect_sp_ops(const rv64_jit_test_generator_t &k) {
    const auto *code = k.getCode<const uint32_t *>();
    const int n_insn = static_cast<int>(k.getSize() / sizeof(uint32_t));

    sp_mem_ops_t ops;
    for (int i = 0; i < n_insn; ++i) {
        if (is_sd_sp(code[i]))
            ops.stores.push_back({sd_rs2(code[i]), sd_imm(code[i])});
        if (is_ld_sp(code[i]))
            ops.loads.push_back({ld_rd(code[i]), ld_imm(code[i])});
    }
    return ops;
}

} // namespace

/// preserve()/restore() pair sd/ld with sp for each callee-saved register
TEST(persistence_preserve_restore, preserve_restore_pairs) {
    const int n_alloc = 10;
    const int n_callee_saved = 3; // n_alloc - 7 caller-saved (t0..t6)

    rv64_jit_probe_t kernel(
            preserve_restore_probe({a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc));
    ASSERT_TRUE(kernel.create_kernel());

    const auto ops = collect_sp_ops(kernel);
    ASSERT_EQ(ops.stores.size(), static_cast<size_t>(n_callee_saved));
    ASSERT_EQ(ops.loads.size(), static_cast<size_t>(n_callee_saved));

    for (int i = 0; i < n_callee_saved; ++i) {
        const auto &store = ops.stores[i];
        const auto &load = ops.loads[i];
        EXPECT_EQ(store.first, load.first) << "register mismatch, slot " << i;
        EXPECT_EQ(store.second, load.second) << "offset mismatch, slot " << i;
    }
}

/// Allocating only caller-saved registers leaves preserve()/restore() no-ops
TEST(persistence_preserve_restore, preserve_noop_caller_saved_only) {
    const int n_alloc = 7; // exactly t0..t6, all caller-saved

    rv64_jit_probe_t kernel(
            preserve_restore_probe({a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc));
    ASSERT_TRUE(kernel.create_kernel());

    const auto ops = collect_sp_ops(kernel);
    EXPECT_EQ(ops.stores.size(), 0u);
    EXPECT_EQ(ops.loads.size(), 0u);
}

// preserve()/restore(): precondition-violation misuse leaves sp balanced

namespace {

// Allocates callee-saved registers, preserves them, then allocates MORE
// callee-saved registers before restoring -- a precondition violation
rv64_jit_probe_t::generate_fn_t preserve_then_overallocate_probe(
        std::vector<Reg> excl, int n_alloc, int n_extra) {
    return [excl, n_alloc, n_extra](rv64_jit_test_generator_t &gen) {
        env_t env(gen,
                excluding_ints(buffer::buffer_t(excl.begin(), excl.end())));
        env.pool.integers().pop(n_alloc);
        env.stack.preserve();
        env.pool.integers().pop(n_extra); // precondition violation
        env.stack.restore();
        gen.ret();
    };
}

// Restores without a preceding preserve() -- a precondition violation
rv64_jit_probe_t::generate_fn_t restore_without_preserve_probe(
        std::vector<Reg> excl, int n_alloc) {
    return [excl, n_alloc](rv64_jit_test_generator_t &gen) {
        env_t env(gen,
                excluding_ints(buffer::buffer_t(excl.begin(), excl.end())));
        env.pool.integers().pop(n_alloc);
        env.stack.restore(); // precondition violation: no matching preserve
        gen.ret();
    };
}

} // namespace

/// restore() pops the count seen at preserve(), not later allocations
TEST(persistence_preserve_restore, restore_ignores_allocation_after_preserve) {
    const int n_alloc = 10; // 3 callee-saved (s0..s2) preserved
    const int n_callee_saved = 3;

    rv64_jit_probe_t kernel(preserve_then_overallocate_probe(
            {a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc, /*n_extra=*/2));
    ASSERT_TRUE(kernel.create_kernel());

    const auto ops = collect_sp_ops(kernel);
    EXPECT_EQ(ops.stores.size(), static_cast<size_t>(n_callee_saved));
    EXPECT_EQ(ops.loads.size(), static_cast<size_t>(n_callee_saved));
}

/// restore() without a preserve() leaves sp alone
TEST(persistence_preserve_restore, restore_without_preserve_is_noop) {
    const int n_alloc = 10; // 3 callee-saved (s0..s2), never preserved

    rv64_jit_probe_t kernel(restore_without_preserve_probe(
            {a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc));
    ASSERT_TRUE(kernel.create_kernel());

    const auto ops = collect_sp_ops(kernel);
    EXPECT_EQ(ops.stores.size(), 0u);
    EXPECT_EQ(ops.loads.size(), 0u);
}

/// preserve() saves the callee-saved registers closed scopes drew
TEST(persistence_preserve_restore, preserve_covers_closed_scopes) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen, excluding_ints({a0, a1, a2, a3, a4, a5, a6, a7}));
        env.pool.scope([&] {
            env.pool.integers().pop(10); // t0..t6, then s0..s2
            return shortfall_t {};
        });
        env.stack.preserve();
        env.stack.restore();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    const auto ops = collect_sp_ops(kernel);
    EXPECT_EQ(ops.stores.size(), 3u);
    EXPECT_EQ(ops.loads.size(), 3u);
}

// stack_t::scope()

/// Later slots reuse closed scopes' slots, and reserve covers the peak
TEST(persistence, scoped_slots_are_reused_and_reserved) {
    int sibling_a = -1, sibling_b = -1, after = -1, reserved = 0;
    rv64_jit_probe_t kernel([&](rv64_jit_test_generator_t &gen) {
        env_t env(gen, excluding_ints({a0}));
        env.stack.new_stack_slot();
        env.stack.scope([&] {
            sibling_a = env.stack.new_stack_slot().offset;
            env.stack.new_stack_slot();
            return shortfall_t {};
        });
        env.stack.scope([&] {
            sibling_b = env.stack.new_stack_slot().offset;
            return shortfall_t {};
        });
        after = env.stack.new_stack_slot().offset;
        const size_t at = gen.getSize();
        env.stack.reserve_stack();
        const auto *code = gen.template getCode<const uint32_t *>();
        reserved = -(static_cast<int32_t>(code[at / sizeof(uint32_t)]) >> 20);
        env.stack.release_stack();
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    EXPECT_EQ(sibling_a, 8);
    EXPECT_EQ(sibling_b, sibling_a);
    EXPECT_EQ(after, sibling_a);
    EXPECT_EQ(reserved, 24);
}

} // namespace rvjit
