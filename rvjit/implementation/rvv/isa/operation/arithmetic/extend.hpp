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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_EXTEND_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_EXTEND_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// Vector integer extension into a type 2, 4 or 8 times as wide
///
/// @details Dispatches to `vsext.vf2/vf4/vf8` on a signed source and to
///     `vzext.vf2/vf4/vf8` on an unsigned one
///
/// @pre The caller must issue `vsetvli` prior to calling this expression and
///     the `vtype` must be configured using the destination's SEW and LMUL
/// @pre The source group is the highest-numbered part of the destination
///     group, or does not overlap it
struct extend_t : public rvjit::arithmetic::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit extend_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        return select_(op) != 0;
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        CHECK_OR_RET(op.is_binary(), , "expected a binary operand descriptor");
        CHECK_OR_RET(all_valid(op.rd, op.rs1) && args.named(2), ,
                "invalid operand descriptor");
        const int sel = select_(op);
        CHECK_OR_RET(sel != 0, , "invalid descriptor");

        const VReg vd(args.rd), vs2(args.rs1);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        switch (sel) {
            case 2: e_->vsext_vf2(vd, vs2, vm); break;
            case 4: e_->vsext_vf4(vd, vs2, vm); break;
            case 8: e_->vsext_vf8(vd, vs2, vm); break;
            case -2: e_->vzext_vf2(vd, vs2, vm); break;
            case -4: e_->vzext_vf4(vd, vs2, vm); break;
            case -8: e_->vzext_vf8(vd, vs2, vm); break;
            default: return;
        }
    }

private:
    rv64::emitter_t e_;

    // The ratio, negative for a zero extension, or 0 when unsupported
    static int select_(const rvjit::arithmetic::op_t &op) {
        if (!op.is_extending_any() || !op.rd.is_vector() || !op.rs1.is_vector())
            return 0;
        if (!op.rd.has_valid_width() || !op.rs1.has_valid_width()) return 0;
        if (!op.rd.is_int() || !op.rs1.is_int()) return 0;

        // A signed source needs a signed destination; an unsigned source
        // extends with zeros into either
        const int ratio = op.rd.width / op.rs1.width;
        if (op.rs1.is_signed()) return op.rd.is_signed() ? ratio : 0;
        return -ratio;
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
