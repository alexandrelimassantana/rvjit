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

#ifndef RVJIT_TEST_ENV_HPP
#define RVJIT_TEST_ENV_HPP

#include <utility>

#include "gtest/gtest.h"

#include "rvjit/implementation/rv64/isa/buffer/stack.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/const_arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/control_flow.hpp"
#include "rvjit/implementation/rv64/isa/operation/transfer/transfer.hpp"
#include "rvjit/implementation/rv64/loop/loopgen.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "rvjit/implementation/rvv/programming_model.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"

namespace rvjit {

/// The components a test builds code with, wired to one code generator
///
/// @note Members are named after the role they play; tests refer to them as
///     `env.pool`, `env.xfer` and so on
struct env_t {
    emitter_t e;
    rv64::const_arithmetic::const_arithmetic_t arith;
    rvv::buffer::architecture_t arch;
    rvv::buffer::buffer_pool_t pool;
    rv64::stack_t stack;
    rv64::transfer::transfer_t xfer;
    rv64::control_flow::control_flow_t cf;
    rv64::loop::loopgen_t loops;

    /// Creates the components over `cg`
    ///
    /// @param cg The code generator instructions are emitted through
    /// @param a The registers the pool draws from
    explicit env_t(codegen_t &cg,
            const rvv::buffer::architecture_t &a
            = rvv::buffer::architecture_t::default_list())
        : e(cg)
        , arith(e)
        , arch(a)
        , pool(arch)
        , stack(e, pool)
        , xfer(e, arith)
        , cf(e, arith, xfer)
        , loops(e, arith, cf, pool, stack, xfer) {}
};

/// An `env_t` extended with the stream components and the RVV model
struct model_env_t : env_t {
    rv64::stream::iterator::iterator_t iterators;
    rvv::memory::memory_t mem;
    stream::memory_operand::memory_operand_t memops;
    rvv::vpu_t vpu;
    rvv::programming_model_t pm;

    /// Creates the components over `cg`
    ///
    /// @param cg The code generator instructions are emitted through
    /// @param a The registers the pool draws from
    /// @param vpu The hardware model
    explicit model_env_t(codegen_t &cg,
            const rvv::buffer::architecture_t &a
            = rvv::buffer::architecture_t::default_list(),
            const rvv::vpu_t &vpu = {})
        : env_t(cg, a)
        , iterators(e, arith, xfer, pool)
        , mem(e)
        , memops(e, iterators, mem, pool)
        , vpu(vpu)
        , pm(e, pool, loops, stack, xfer, arith, cf, memops, this->vpu) {}
};

/// The private code generator a `hosted_t` emits into
struct code_holder_t {
    codegen_t cg;
};

/// An environment over a private code generator, for tests that execute nothing
template <typename Env>
struct hosted_t : code_holder_t, Env {
    /// Creates the environment over a private code generator
    ///
    /// @param args The arguments after the code generator
    template <typename... Args>
    explicit hosted_t(Args &&...args) : Env(cg, std::forward<Args>(args)...) {}
};

using host_env_t = hosted_t<env_t>;
using host_model_env_t = hosted_t<model_env_t>;

/// gtest fixture exposing a buffer pool over the default registers
struct pool_test_t : ::testing::Test {
    rvv::buffer::architecture_t arch
            = rvv::buffer::architecture_t::default_list();
    rvv::buffer::buffer_pool_t pool {arch};
};

} // namespace rvjit

#endif
