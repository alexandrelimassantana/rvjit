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

#ifndef RVJIT_TEST_RVV_STREAM_MEMORY_MEMORY_OPERAND_HELPERS_HPP
#define RVJIT_TEST_RVV_STREAM_MEMORY_MEMORY_OPERAND_HELPERS_HPP

#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;
using rv64_iterator_t = rvjit::rv64::stream::iterator::iterator_t;
using memory_operand_t = rvjit::stream::memory_operand::memory_operand_t;
using memory_spec_t = rvjit::stream::memory_operand::spec_t;
using memory_variant_t = rvjit::stream::memory_operand::variant_t;

inline memory_spec_t make_spec(const memory::op_t &op, const pointer_t &origin,
        const space_t &space, addressing_t mode = addressing_t::base_simm,
        update_t end = update_t::advance(0)) {
    memory_spec_t spec;
    spec.op = op;
    spec.origin = origin;
    spec.space = space;
    spec.addressing = mode;
    spec.update = end;
    return spec;
}

inline memory_variant_t make_variant(strategy_id_t strategy, int ntmp,
        const buffer_map_t &ring = buffer_map_t({1, 1})) {
    memory_variant_t variant;
    variant.strategy = strategy;
    variant.ntmp = ntmp;
    variant.ring = ring;
    return variant;
}

} // namespace rvjit

#endif
