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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/bitcast.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/zero.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// RV64 arithmetic dialect: scalar zero and bitcast only
struct arithmetic_t : public operation::arithmetic_t {
    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit arithmetic_t(rvjit::emitter_t e)
        : zero_impl_(e), bitcast_impl_(e) {}

protected:
    const rvjit::arithmetic::expression_impl_t &zero_() const override {
        return zero_impl_;
    }
    const rvjit::arithmetic::expression_impl_t &bitcast_() const override {
        return bitcast_impl_;
    }

    zero_t zero_impl_;
    bitcast_t bitcast_impl_;
};

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#endif
