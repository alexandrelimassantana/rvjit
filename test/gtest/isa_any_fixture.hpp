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

#ifndef RVJIT_TEST_ISA_ANY_FIXTURE_HPP
#define RVJIT_TEST_ISA_ANY_FIXTURE_HPP

#include "gtest/gtest.h"

#include "rvjit/implementation/common/isa/buffer/allocator/group.hpp"
#include "rvjit/implementation/common/isa/buffer/allocator/named.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"

namespace rvjit {

/// The names `first` to `last`, both included
inline buffer::buffer_t name_range(int first, int last) {
    buffer::buffer_t b;
    for (int n = first; n <= last; ++n)
        b.set(static_cast<name_t>(n));
    return b;
}

/// An architecture of plain names, with no ISA behind them
///
/// @note The integers skip name 0, which registers read as absent
inline buffer::architecture_t isa_any_architecture() {
    buffer::architecture_t a;
    a.integer = buffer::buffer_file_t(name_range(5, 31));
    a.floating = buffer::buffer_file_t(name_range(0, 19));
    a.vector = buffer::buffer_file_t(name_range(0, 31));
    return a;
}

/// A buffer pool over `isa_any_architecture()`
using isa_any_pool_t = buffer::buffer_compose_t<buffer::allocator::named_t,
        buffer::allocator::named_t, buffer::allocator::group_t>;

/// A buffer pool for any ISA, with the architecture it draws from
struct isa_any_env_t {
    buffer::architecture_t arch = isa_any_architecture();
    isa_any_pool_t pool {arch};
};

/// gtest fixture exposing a buffer pool for any ISA
struct isa_any_pool_test_t : ::testing::Test {
    buffer::architecture_t arch = isa_any_architecture();
    isa_any_pool_t pool {arch};
};

} // namespace rvjit

#endif
