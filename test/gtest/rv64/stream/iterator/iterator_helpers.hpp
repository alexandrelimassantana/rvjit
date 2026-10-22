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

#ifndef RVJIT_TEST_RV64_STREAM_ITERATOR_ITERATOR_HELPERS_HPP
#define RVJIT_TEST_RV64_STREAM_ITERATOR_ITERATOR_HELPERS_HPP

#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/rv64/stream/iterator/strategies/pivot.hpp"
#include "rvjit/interface/isa/const.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;
using rv64_iterator_t = rvjit::rv64::stream::iterator::iterator_t;

/// Assigns `ap.ngpr` scratch registers (from `ids`) to `ap`, if it needs any
inline void assign_regs(axis_plan_t &ap, const name_t *ids) {
    if (ap.ngpr > 0) ap.regs = buffer::buffer_t(ids, ap.ngpr);
}

/// Builds a plan() request from its fields, defaulting to
/// today's most common combination (any strategy, immediate addressing,
/// advance(0) end-state)
inline plan_t make_request(const pointer_t &origin, const space_t &space,
        strategy_id_t strategy = strategy_id_t::any(),
        addressing_t mode = addressing_t::base_simm,
        update_t end = update_t::advance()) {
    plan_t p;
    p.origin = origin;
    p.space = space;
    p.strategy = strategy;
    p.addressing = mode;
    p.update = end;
    return p;
}

/// Stores `p.base + p.offset` at `out_ptr + byte_offset`
///
/// @note Materializes the address into `value_tmp`/`addr_tmp` first, since
///     `p.offset` may be nonzero and `byte_offset` may exceed simm12
inline void store_pointer(Xbyak_riscv::CodeGenerator &cg, const pointer_t &p,
        const Reg &value_tmp, const Reg &addr_tmp, const Reg &out_ptr,
        int64_t byte_offset) {
    Reg value_reg = p.base;
    if (p.offset != 0) {
        cg.addi(value_tmp, p.base, p.offset);
        value_reg = value_tmp;
    }
    if (byte_offset >= -2048 && byte_offset <= 2047) {
        cg.sd(value_reg, out_ptr, static_cast<int>(byte_offset));
    } else {
        cg.li(addr_tmp, static_cast<uint32_t>(byte_offset));
        cg.add(addr_tmp, addr_tmp, out_ptr);
        cg.sd(value_reg, addr_tmp, 0);
    }
}

/// Checks that out_addrs[j] equals base_value + j * stride for every j
inline void expect_address_progression(const std::vector<int64_t> &out_addrs,
        int64_t base_value, int64_t stride) {
    for (size_t j = 0; j < out_addrs.size(); ++j) {
        const int64_t want = base_value + static_cast<int64_t>(j) * stride;
        ASSERT_EQ(out_addrs[j], want) << "at j=" << j;
    }
}

} // namespace rvjit

#endif
