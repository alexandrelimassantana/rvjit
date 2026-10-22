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

#ifndef RVJIT_TEST_LOOP_MOCK_HPP
#define RVJIT_TEST_LOOP_MOCK_HPP

#include <vector>
#include <initializer_list>

#include "rvjit/interface/loop/loop.hpp"

namespace rvjit {

/// A loop that calls its body once per configured stage width
struct mock_loop_t : public loop::loop_t {
    /// Creates a loop of the given stages
    ///
    /// @param stages The unroll factor each stage dispatches with, in order
    explicit mock_loop_t(std::initializer_list<int> stages) : stages_(stages) {}

    bool generate(const control_flow::on_index_t &cb) override {
        for (int u : stages_)
            cb(u);
        return true;
    }

    bool is_valid() const override { return true; }

private:
    std::vector<int> stages_;
};

} // namespace rvjit

#endif
