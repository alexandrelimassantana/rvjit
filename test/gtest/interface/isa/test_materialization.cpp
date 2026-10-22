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

#include "isa_any_fixture.hpp"
#include "isa_any_registers.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/variable.hpp"

namespace rvjit {

using namespace isa_any;

namespace {

/// Needs a fixed number of buffers
struct wants_t : buffered_t {
    explicit wants_t(int n) : n_(n) {}
    int buffer_needed() const override {
        const int left = n_ - buffer_.size();
        return left > 0 ? left : 0;
    }
    const buffer::buffer_t &buffer() const { return buffer_; }

private:
    int n_;
};

} // namespace

// variable_t: field population, no code is executed

/// `copy` sets a variable's reload origin, unlike `adopt`, its own register
TEST(transfer_make_value, copy_is_a_reload_origin) {
    const variable_t rv(r10, source_t::copy(r10));
    EXPECT_TRUE(rv.buffer() == r10);
    EXPECT_TRUE(rv.needs_to_materialize());
}

/// An overflowing immediate keeps its source whatever register is given
TEST(transfer_make_value, imm_with_reg_is_immediate) {
    isa_any_env_t env;

    variable_t rv = env.pool.new_variable(source_t::immediate(2048), r10);
    EXPECT_TRUE(rv.buffer() == r10);
    EXPECT_TRUE(rv.needs_to_materialize());
}

/// A plain immediate is lazy: not materializable until a buffer is assigned
TEST(transfer_make_value, plain_imm_is_lazy) {
    variable_t rv(source_t::immediate(42));
    EXPECT_TRUE(rv.needs_buffer());
    EXPECT_FALSE(rv.source().is_none());
    EXPECT_FALSE(rv.needs_to_materialize());
}

// buffer_pool_t::allocate(variable_t&)

/// A materialized variable_t passes through allocate() with no allocation
TEST(transfer_new, allocate_variable_eager_passthrough) {
    isa_any_env_t env;
    const reg_t r = reg_t(env.pool.integers().pop());
    const int before = env.pool.integers().available();
    variable_t rv(r);
    env.pool.allocate(rv);
    EXPECT_EQ(before - env.pool.integers().available(), 0);
    EXPECT_TRUE(rv.buffer() == r);
}

/// A loadable variable_t without a register gets one, keeping its source
TEST(transfer_new, allocate_variable_lazy_materializes) {
    isa_any_env_t env;
    const int before = env.pool.integers().available();
    variable_t rv {variable_t(source_t::immediate(5))};
    env.pool.allocate(rv);
    EXPECT_EQ(before - env.pool.integers().available(), 1);
    EXPECT_TRUE(is_not_zero(rv.buffer()));
    EXPECT_TRUE(rv.needs_to_materialize());
}

/// allocate() leaves `source()` unchanged and only touches the register
TEST(transfer_new, allocate_variable_preserves_source) {
    isa_any_env_t env;
    const reg_t buffer_reg = reg_t(env.pool.integers().pop());

    variable_t input = variable_t(source_t::copy(buffer_reg));

    env.pool.allocate(input);
    EXPECT_TRUE(is_not_zero(input.buffer()));
    ASSERT_TRUE(input.source().is_copy());
    EXPECT_TRUE(input.source().buffer() == buffer_reg);
}

/// allocate() draws the registers an object needs, and none once it has them
TEST(transfer_new, allocate_draws_every_register_the_object_needs) {
    isa_any_env_t env;
    const int before = env.pool.integers().available();
    wants_t single(1), multi(3);
    ASSERT_TRUE(env.pool.allocate(single));
    ASSERT_TRUE(env.pool.allocate(multi));
    EXPECT_EQ(single.buffer().size(), 1);
    EXPECT_EQ(multi.buffer().size(), 3);
    EXPECT_EQ(before - env.pool.integers().available(), 4);

    ASSERT_TRUE(env.pool.allocate(multi));
    EXPECT_EQ(before - env.pool.integers().available(), 4);
}

/// new_buffer_source() on an exhausted pool is not valid
TEST(transfer_new, new_buffer_source_on_an_exhausted_pool_is_invalid) {
    isa_any_env_t env;
    env.pool.integers().pop(env.pool.integers().available());
    EXPECT_FALSE(env.pool.new_buffer_source().is_valid());
}

/// A variable never stores an `adopt` origin: it is kept as a `copy`
TEST(transfer_new, variable_stores_adopt_as_copy) {
    const variable_t v(r11, source_t::adopt(r10));
    EXPECT_TRUE(v.source().is_copy());
}

/// A variable materializes given a lazy origin and a buffer
TEST(transfer_new, variable_need_follows_lazy_origin_and_buffer) {

    EXPECT_FALSE(variable_t(r10).needs_to_materialize());
    EXPECT_TRUE(variable_t(r10).is_valid());
    EXPECT_FALSE(variable_t().is_valid());

    EXPECT_TRUE(variable_t(r10, source_t::immediate(5)).needs_to_materialize());
    EXPECT_TRUE(variable_t(r10, source_t::copy(r11)).needs_to_materialize());
    {
        const variable_t v(r10, source_t::load(pointer_t(r11, 0)));
        EXPECT_TRUE(v.needs_to_materialize());
    }

    // Without a buffer there is nowhere to materialize into
    EXPECT_FALSE(variable_t(source_t::immediate(5)).needs_to_materialize());
}

// step_t::macc: buffer needs, no code is executed

/// A macc whose product is known and fits simm12 needs no scratch register
TEST(transfer_make_value, macc_known_product_needs_no_buffer) {
    const auto step = step_t::macc(const_t(3), const_t(4));
    EXPECT_FALSE(step.macc_needs_multiply());
    EXPECT_FALSE(step.needs_buffer());
}

/// A macc needing a real multiply wants a scratch register until assigned
TEST(transfer_make_value, macc_needing_multiply_needs_buffer_until_assigned) {
    isa_any_env_t env;
    const reg_t lhs_reg = reg_t(env.pool.integers().pop());

    step_t step = step_t::macc(const_t(lhs_reg), const_t(4));
    EXPECT_TRUE(step.macc_needs_multiply());
    EXPECT_TRUE(step.needs_buffer());

    step.set_buffer(reg_t(env.pool.integers().pop()));
    EXPECT_FALSE(step.needs_buffer());
}

/// An add step never claims a buffer; is_valid() reports an unfulfilled one
TEST(transfer_make_value, add_step_never_needs_buffer) {
    step_t step = step_t::plain(const_t(100000));
    EXPECT_FALSE(step.needs_buffer());
    EXPECT_FALSE(step.has_buffer());
    EXPECT_FALSE(step.is_valid());

    step = step_t::plain(const_t(100000, r10));
    EXPECT_FALSE(step.needs_buffer());
    EXPECT_TRUE(step.is_valid());
}

/// A macc scratch is the step's own buffer; operand registers stay untouched
TEST(transfer_make_value, macc_buffer_is_scratch_only) {
    step_t step = step_t::macc(const_t(r10), const_t(4));
    ASSERT_TRUE(step.needs_buffer());
    step.set_buffer(r11);
    EXPECT_FALSE(step.needs_buffer());
    EXPECT_TRUE(step.buffer() == r11);
    EXPECT_TRUE(step.src1.buffer() == r10);
}

/// A `sub` step never claims a buffer and is valid once its constant is
TEST(transfer_make_value, sub_step_never_needs_buffer) {
    step_t step = step_t::sub(const_t(100000));
    EXPECT_TRUE(step.is_sub());
    EXPECT_FALSE(step.needs_buffer());
    EXPECT_FALSE(step.is_valid());

    step = step_t::sub(const_t(r10));
    EXPECT_TRUE(step.is_valid());
}

// new_variable(source_t::copy): the caller's register is never adopted

/// A `copy` source gets its own allocated primary register, filled lazily
TEST(transfer_new, new_variable_from_copy_allocates_primary) {
    isa_any_env_t env;
    const reg_t buffer = reg_t(env.pool.integers().pop());
    const int before = env.pool.integers().available();
    const variable_t rv = env.pool.new_variable(source_t::copy(buffer));
    EXPECT_EQ(before - env.pool.integers().available(), 1);
    EXPECT_TRUE(is_not_zero(rv.buffer()));
    EXPECT_TRUE(rv.buffer() != buffer);
    EXPECT_TRUE(rv.needs_to_materialize());
}

// new_variable(source_t::adopt): the register becomes the variable's buffer

/// An `adopt` source is taken directly, with no allocation
TEST(transfer_new, new_variable_from_adopt_takes_register) {
    isa_any_env_t env;
    const reg_t buffer = reg_t(env.pool.integers().pop());
    const int before = env.pool.integers().available();
    const variable_t rv = env.pool.new_variable(source_t::adopt(buffer));
    EXPECT_EQ(before - env.pool.integers().available(), 0);
    EXPECT_TRUE(rv.buffer() == buffer);
    EXPECT_FALSE(rv.needs_to_materialize());
}

// buffer_pool_t::new_variable(source_t::load)

/// A pointer-backed variable_t allocates exactly one primary register
TEST(transfer_new, new_variable_from_pointer_allocates_register) {
    isa_any_env_t env;
    const int before = env.pool.integers().available();
    const variable_t rv
            = env.pool.new_variable(source_t::load(pointer_t(stack, 0)));
    EXPECT_EQ(before - env.pool.integers().available(), 1);
    EXPECT_TRUE(is_not_zero(rv.buffer()));
    EXPECT_TRUE(rv.source().is_load());
}

// buffer_pool_t::new_constant(source_t)

/// new_constant() allocates only what the constant still needs
TEST(transfer_new, new_constant_allocates_only_what_it_needs) {
    isa_any_env_t env;
    const reg_t held = reg_t(env.pool.integers().pop());
    const auto drawn = [&](const source_t &src) {
        const int before = env.pool.integers().available();
        env.pool.new_constant(src);
        return before - env.pool.integers().available();
    };
    EXPECT_EQ(drawn(source_t::immediate(42)), 0);
    EXPECT_EQ(drawn(source_t::immediate(1 << 20)), 1);
    EXPECT_EQ(drawn(source_t::adopt(held)), 0);
    EXPECT_EQ(drawn(source_t::copy(held)), 0);
    EXPECT_EQ(drawn(source_t::load(pointer_t(stack, 0))), 1);
    EXPECT_EQ(drawn(source_t()), 0);
    EXPECT_FALSE(env.pool.new_constant(source_t()).is_valid());
}

// materializable_t: state is derived on demand from the object's own fields

/// A direct edit of `const_t::kind` shows in the next query
TEST(materializable, const_kind_edit_is_seen_immediately) {
    const_t c(5);
    EXPECT_FALSE(c.needs_buffer());
    EXPECT_FALSE(c.needs_to_materialize());

    c.kind = const_t::kind_t::imm_reg;
    EXPECT_TRUE(c.needs_buffer());
    EXPECT_TRUE(c.needs_to_materialize());
    EXPECT_FALSE(c.is_valid());

    c.set_buffer(r10);
    EXPECT_FALSE(c.needs_buffer());
    EXPECT_TRUE(c.is_valid());
}

/// Editing `step_t::src2` to overflow simm12 makes the step ask for a scratch
TEST(materializable, step_operand_edit_is_seen_immediately) {
    step_t s = step_t::macc(const_t(3), const_t(4));
    EXPECT_FALSE(s.needs_buffer());

    s.src2 = const_t(1 << 20, r12);
    EXPECT_TRUE(s.needs_buffer());
    EXPECT_FALSE(s.is_valid());

    s.set_buffer(r11);
    EXPECT_FALSE(s.needs_buffer());
    EXPECT_TRUE(s.is_valid());
}

/// A query through the base type reaches the derived implementation
TEST(materializable, base_reference_dispatches_to_the_derived_type) {
    const_t c = const_t::invalid();
    const materializable_t &base = c;
    EXPECT_FALSE(base.is_valid());

    const variable_t v(r10, source_t::immediate(3));
    const materializable_t &vbase = v;
    EXPECT_TRUE(vbase.needs_to_materialize());
    EXPECT_TRUE(vbase.demand().is_none());

    const variable_t pending(source_t::immediate(3));
    const materializable_t &pbase = pending;
    EXPECT_EQ(pbase.demand()[resource_t::integers], 1);
}

// const_t(source_t): the constant resolved from each source kind

TEST(const_from_source, immediate_fits_or_needs_a_register) {
    const const_t small(source_t::immediate(42));
    EXPECT_TRUE(small.is_imm());
    EXPECT_EQ(small.imm(), 42);

    const const_t big(source_t::immediate(1 << 20));
    EXPECT_TRUE(big.is_imm_reg());
    EXPECT_TRUE(big.needs_buffer());
}

/// `adopt` and `copy` registers are both read in place
TEST(const_from_source, registers_are_read_in_place) {
    for (const source_t &src : {source_t::adopt(r10), source_t::copy(r10)}) {
        const const_t c(src);
        EXPECT_TRUE(c.is_reg());
        EXPECT_TRUE(c.buffer() == r10);
        EXPECT_FALSE(c.needs_buffer());
        EXPECT_FALSE(c.needs_to_materialize());
        EXPECT_TRUE(c.is_valid());
    }
}

TEST(const_from_source, load_needs_a_register_and_a_load) {
    const const_t c(source_t::load(pointer_t(stack, 8)));
    EXPECT_TRUE(c.is_loaded());
    EXPECT_TRUE(c.needs_buffer());
    EXPECT_TRUE(c.needs_to_materialize());
    EXPECT_TRUE(c.source().memory().base == stack);
}

TEST(const_from_source, none_and_invalid_sources_are_invalid) {
    EXPECT_FALSE(const_t(source_t()).is_valid());
    EXPECT_FALSE(const_t(source_t::adopt(reg_t())).is_valid());
}

} // namespace rvjit
