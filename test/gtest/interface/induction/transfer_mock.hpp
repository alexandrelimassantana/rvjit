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

#ifndef RVJIT_TEST_INTERFACE_INDUCTION_TRANSFER_MOCK_HPP
#define RVJIT_TEST_INTERFACE_INDUCTION_TRANSFER_MOCK_HPP

#include <string>
#include <utility>
#include <vector>

#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// A transfer component that records its calls instead of emitting code
///
/// @details Each call becomes one CSV row, `verb,mode,dest,src1,src2`:
///     - verbs: `get`, `put`, `advance`, and any row added by `mark()`
///     - source and slot modes: `reg`, `imm`, `mem`
///     - step modes: `imm`, `reg`, `sub_imm`, `sub_reg`, `macc`
///     - operands: a register label (see `name()`), an immediate, `base+offset`
struct recording_transfer_t : public operation::transfer_t {
    using operation::transfer_t::advance;

    void get(const reg_t &dst, const source_t &src) const override {
        const bool is_imm = src.is_immediate();
        const bool is_mem = src.is_load();
        const std::string mode = is_imm ? "imm" : is_mem ? "mem" : "reg";
        const std::string from = is_imm ? std::to_string(src.immediate())
                : is_mem                ? mem_(src.memory())
                                        : reg_(src.buffer());
        csv_ += "get," + mode + "," + reg_(dst) + "," + from + "\n";
    }

    void put(const slot_t &slot, const reg_t &value) const override {
        const bool is_mem = slot.is_memory();
        const std::string to
                = is_mem ? mem_(slot.memory()) : reg_(slot.buffer());
        csv_ += std::string("put,") + (is_mem ? "mem," : "reg,") + to + ","
                + reg_(value) + "\n";
    }

    void advance(const reg_t &rd, const reg_t &rs,
            const step_t &step) const override {
        std::string mode = "none", by;
        if (step.is_macc()) {
            mode = "macc";
            by = const_(step.src1) + "*" + const_(step.src2);
        } else if (!step.is_none()) {
            mode = std::string(step.is_sub() ? "sub_" : "")
                    + (step.src1.has_reg() ? "reg" : "imm");
            by = const_(step.src1);
        }
        csv_ += "advance," + mode + "," + reg_(rd) + "," + reg_(rs) + "," + by
                + "\n";
    }

    /// Adds a row, for the code a test runs between the calls
    ///
    /// @param what The row
    void mark(const std::string &what) const { csv_ += what + "\n"; }

    /// Prints register `r` as `label` in the rows
    ///
    /// @param r The register
    /// @param label The name printed
    void name(const reg_t &r, const std::string &label) {
        labels_.push_back({reg_name(r), label});
    }

    /// The rows recorded so far, one per line
    const std::string &csv() const { return csv_; }

private:
    std::string reg_(const reg_t &r) const {
        for (const auto &l : labels_)
            if (l.first == reg_name(r)) return l.second;
        return "x" + std::to_string(static_cast<int>(reg_name(r)));
    }

    std::string mem_(const pointer_t &p) const {
        return reg_(p.base) + "+" + std::to_string(p.offset);
    }

    std::string const_(const const_t &c) const {
        return c.has_reg() ? reg_(c.buffer()) : std::to_string(c.imm());
    }

    mutable std::string csv_;
    std::vector<std::pair<name_t, std::string>> labels_;
};

} // namespace rvjit

#endif
