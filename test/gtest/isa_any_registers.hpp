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

#ifndef RVJIT_TEST_ISA_ANY_REGISTERS_HPP
#define RVJIT_TEST_ISA_ANY_REGISTERS_HPP

#include "rvjit/interface/isa/register.hpp"

namespace rvjit {
namespace isa_any {

// Plain register names, with no ISA behind them
constexpr reg_t r5 {5}, r6 {6}, r7 {7}, r8 {8}, r9 {9}, r10 {10}, r11 {11},
        r12 {12}, r13 {13}, r14 {14}, r15 {15}, r16 {16}, r17 {17}, r18 {18},
        r19 {19}, r20 {20}, r21 {21}, r22 {22}, r23 {23}, r24 {24}, r25 {25},
        r26 {26}, r27 {27}, r28 {28}, r29 {29}, r30 {30}, r31 {31};

constexpr reg_t zero = reg_t::zero();
constexpr reg_t stack = reg_t::stack();

} // namespace isa_any
} // namespace rvjit

#endif
