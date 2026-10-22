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

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/variable.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

// transfer_t::init

class transfer_materialize_t
    : public ::testing::TestWithParam<std::tuple<int64_t>> {};

/// init() emits initialization code only when the value overflows simm12
TEST_P(transfer_materialize_t, emit_check) {
    int c;
    std::tie(c) = GetParam();

    rv64_jit_probe_t kernel([c](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        env.xfer.init({const_t(c, Xbyak_riscv::a0)});
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());

    // Check if more than 1 instruction (ret) was emitted
    const bool should_emit = is_simm12(c) == false;
    const bool emitted_init = kernel.getSize() > sizeof(int32_t);

    ASSERT_EQ(should_emit, emitted_init);
}

/// Cases that result in no-op
INSTANTIATE_TEST_SUITE_P(noop, transfer_materialize_t,
        ::testing::Values(0, 16, -16, 2047, -2048));

/// Cases that result in code emission
INSTANTIATE_TEST_SUITE_P(emit, transfer_materialize_t,
        ::testing::Values(2048, 2049, -2049, -2050));

// new_variable(source_t::immediate): lazily allocated and loaded

class transfer_make_value_correctness_t
    : public ::testing::TestWithParam<std::tuple<int64_t>> {};

/// A lazily allocated variable_t materializes its constant once loaded
TEST_P(transfer_make_value_correctness_t, matches_value) {
    int c;
    std::tie(c) = GetParam();

    rv64_jit_probe_t kernel([c](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg ptr = Xbyak_riscv::a0;

        variable_t rv = env.pool.new_variable(source_t::immediate(c));
        env.xfer.init(rv);

        gen.sd(rv.buffer(), ptr);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t received;
    kernel(&received);

    ASSERT_EQ(c, received);
}

// A simm12-fitting value has no register of its own yet
INSTANTIATE_TEST_SUITE_P(simm12, transfer_make_value_correctness_t,
        ::testing::Values(0, 42, -2048, 2047));

// An overflowing value also starts without a register
INSTANTIATE_TEST_SUITE_P(overflow, transfer_make_value_correctness_t,
        ::testing::Values(2048, -2049, 1 << 20));

/// `source_t::immediate` re-materializes correctly on every init()
TEST(transfer_make_value, immediate_rematerializes_on_every_init) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);

        const Reg ptr = Xbyak_riscv::a0;
        const Reg r = Xbyak_riscv::a1;

        variable_t rv = env.pool.new_variable(source_t::immediate(2048), r);
        env.xfer.init(rv);

        gen.sd(rv.buffer(), ptr);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t received;
    kernel(&received);

    ASSERT_EQ(2048, received);
}

// transfer_t::advance(step_t::macc)

/// A known product that fits simm12 folds into a single `add`
TEST(transfer, macc_advance_both_imm_fits_simm12) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, 10);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::macc(const_t(3), const_t(4));
        env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 10 + 3 * 4);
}

/// A known product overflowing simm12 is materialized in the scratch register
TEST(transfer, macc_advance_both_imm_overflow) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        const Reg tmp = Reg(env.pool.integers().pop());
        gen.li(r, 10);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::macc(const_t(100000), const_t(1000), tmp);
        env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 10 + int64_t(100000) * 1000);
}

/// A register multiplicand needs a runtime multiply into the scratch register
TEST(transfer, macc_advance_register_operand) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        const Reg lhs_reg = Reg(env.pool.integers().pop());
        const Reg tmp = Reg(env.pool.integers().pop());
        gen.li(r, 10);
        gen.li(lhs_reg, 7);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::macc(const_t(lhs_reg), const_t(4), tmp);
        env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 10 + 7 * 4);
}

// new_variable(source_t::load)

/// Builds a probe that loads a source_t::memory variable_t from `offset`
/// bytes into `src_ptr` and writes it to `out_ptr`
rv64_jit_probe_t::generate_fn_t make_value_pointer_probe(int offset) {
    return [offset](rv64_jit_test_generator_t &gen) {
        const Reg src_ptr = Xbyak_riscv::a0;
        const Reg out_ptr = Xbyak_riscv::a1;
        env_t env(gen, excluding_ints({src_ptr, out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());

        variable_t v = env.pool.new_variable(
                source_t::load(pointer_t(src_ptr, offset)), r);
        env.xfer.init(v);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    };
}

/// source_t::memory with a zero offset loads exactly the pointer's base
TEST(transfer, make_value_pointer_loads_base_only) {
    rv64_jit_probe_t kernel(make_value_pointer_probe(0));
    ASSERT_TRUE(kernel.create_kernel());
    int64_t src = 777;
    int64_t out = 0;
    kernel(&src, &out);
    ASSERT_EQ(out, 777);
}

/// source_t::memory honors a nonzero offset into the pointer's base
TEST(transfer, make_value_pointer_honors_nonzero_offset) {
    rv64_jit_probe_t kernel(make_value_pointer_probe(8));
    ASSERT_TRUE(kernel.create_kernel());
    const int64_t src[2] = {111, 222};
    int64_t out = 0;
    kernel(src, &out);
    ASSERT_EQ(out, 222);
}

/// source_t::memory works as a binding origin, recovering a clobbered register
TEST(transfer, make_value_pointer_recovered_across_body) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg src_ptr = Xbyak_riscv::a0;
        const Reg out_ptr = Xbyak_riscv::a1;
        env_t env(gen, excluding_ints({src_ptr, out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());

        const variable_t v
                = env.pool.new_variable(source_t::load(pointer_t(src_ptr)), r);
        rvjit::bind_t entry;
        entry.reg = r;
        entry.origin = v.source();

        // Clobber r inside body -- restore() must recover it afterward;
        // preserve() is a no-op (no slot)
        env.xfer.preserve(entry);
        gen.li(r, -1);
        env.xfer.restore(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t src = 777;
    int64_t out = 0;
    kernel(&src, &out);
    ASSERT_EQ(out, 777);
}

/// A binding with only a step is stepped in place by advance()
TEST(transfer, plain_reg_direct_step) {
    static const int64_t start = 10;
    static const int delta = 3;
    static const int passes = 4;

    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, start);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::plain(const_t(delta));
        for (int i = 0; i < passes; ++i)
            env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 10 + 3 * 4);
}

/// A binding on a register slot saves a clobbered primary register into it
TEST(transfer, register_slot_recovers_across_body) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg a = Reg(env.pool.integers().pop());
        const Reg b = Reg(env.pool.integers().pop());
        gen.li(a, 42);

        rvjit::bind_t entry;
        entry.reg = a;
        entry.slot = slot_t(b);
        env.xfer.preserve(entry);
        gen.li(a, -1); // clobber a -- must be recovered from b
        env.xfer.restore(entry);

        gen.sd(a, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 42);
}

// transfer_t::advance(step_t::sub)

/// A register operand is subtracted through the register itself
TEST(transfer, sub_advance_register_operand) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        const Reg amount = Reg(env.pool.integers().pop());
        gen.li(r, 100);
        gen.li(amount, 7);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::sub(const_t(amount));
        env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 100 - 7);
}

/// A known immediate is negated into a single `add`
TEST(transfer, sub_advance_immediate_operand) {
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        const Reg out_ptr = Xbyak_riscv::a0;
        env_t env(gen, excluding_ints({out_ptr}));
        const Reg r = Reg(env.pool.integers().pop());
        gen.li(r, 100);

        rvjit::bind_t entry;
        entry.reg = r;
        entry.step = step_t::sub(const_t(30));
        env.xfer.advance(entry);

        gen.sd(r, out_ptr, 0);
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    int64_t out = 0;
    kernel(&out);
    ASSERT_EQ(out, 100 - 30);
}

} // namespace rvjit
