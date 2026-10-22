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

#include "isa_any_registers.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

using namespace isa_any;

TEST(all_valid, plain_objects) {
    const const_t imm(4);
    const const_t big(1 << 20);
    EXPECT_TRUE(all_valid(imm, imm));
    EXPECT_FALSE(all_valid(imm, big));
}

} // namespace rvjit
