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

#include "gtest/gtest.h"

#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// An explicit int set draws candidates in the register file's allocation order
TEST(buffer_source, int_register_file_visible_preserves_order) {
    const Reg ints[] = {a2, t0, s1};
    rvv::buffer::architecture_t source(buffer::buffer_t(ints, ints + 3));
    rvv::buffer::buffer_pool_t pool(source);
    EXPECT_EQ(pool.integers().pop(), reg_name(t0));
    EXPECT_EQ(pool.integers().pop(), reg_name(a2));
    EXPECT_EQ(pool.integers().pop(), reg_name(s1));
}

/// Listing order doesn't matter: callee-saved follow caller-saved registers
TEST(buffer_source, int_register_file_visible_reorders_callee_saved) {
    const Reg ints[] = {s1, a2, t0};
    rvv::buffer::architecture_t source(buffer::buffer_t(ints, ints + 3));
    rvv::buffer::buffer_pool_t pool(source);
    EXPECT_EQ(pool.integers().pop(), reg_name(t0));
    EXPECT_EQ(pool.integers().pop(), reg_name(a2));
    EXPECT_EQ(pool.integers().pop(), reg_name(s1));
}

// set_vector_group_size(m_vec, m_acc): a rvv::buffer::architecture_t constructed
// with an explicit vector list has each pool keep only the ids from that
// list that are valid group starts for its own LMUL

namespace {

// v2 is a valid m2 start but not m4; v8 is valid for both
rvv::buffer::architecture_t explicit_vector_source() {
    const VReg vecs[] = {v2, v8};
    return rvv::buffer::architecture_t(
            {}, {}, buffer::buffer_t(vecs, vecs + 2));
}

} // namespace

/// The input pool keeps every source id that is a valid m2 group start
TEST(buffer_source, widening_visible_keeps_valid_inputs) {
    rvv::buffer::architecture_t source = explicit_vector_source();
    rvv::buffer::buffer_pool_t pool(source);
    pool.set_vector_group_size(vgroup_size(LMUL::m2), vgroup_size(LMUL::m4));
    EXPECT_EQ(pool.vectors().pop(), reg_name(v2));
    EXPECT_EQ(pool.vectors().pop(), reg_name(v8));
}

/// The accumulator pool drops source ids that aren't valid m4 group starts
TEST(buffer_source, widening_visible_filters_invalid_acc) {
    // v2 is a valid m2 start but not m4
    rvv::buffer::architecture_t source = explicit_vector_source();
    rvv::buffer::buffer_pool_t pool(source);
    pool.set_vector_group_size(vgroup_size(LMUL::m2), vgroup_size(LMUL::m4));
    EXPECT_EQ(pool.vectors().pop(buffer::allocation_t::output), reg_name(v8));
}

} // namespace rvjit
