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

#ifndef RVJIT_TEST_STREAM_MOCK_HPP
#define RVJIT_TEST_STREAM_MOCK_HPP

#include <deque>
#include <functional>
#include <vector>
#include <initializer_list>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

namespace rvjit {

/// Owns streams running caller-given callbacks at each position
struct mock_streams_t {
    using on_pos_t = std::function<void(const index_t &)>;

    /// Creates a stream of the given shape
    stream::ref_t add(std::initializer_list<int> shape, on_pos_t produce = {},
            on_pos_t init = {}) {
        entries_.emplace_back();
        entry_t &e = entries_.back();
        e.nd = static_cast<int>(shape.size());
        int d = 0;
        for (int s : shape)
            e.shape[d++] = s;
        e.produce = std::move(produce);
        e.init = std::move(init);
        return e.ref();
    }

private:
    struct entry_t : public stream::stream_impl_t {
        int ndims() const override { return nd; }
        int size(int d) const override { return shape[d]; }
        bool is_valid() const override { return true; }
        bool resize(const index_t &s) override {
            shape = s;
            return true;
        }

        int nd = 0;
        index_t shape {};
        on_pos_t produce, init;

    protected:
        void produce_(const index_t &idx) override {
            if (produce) produce(idx);
        }
        void init_(const index_t &idx) override {
            if (init) init(idx);
        }
    };
    std::deque<entry_t> entries_;
};

} // namespace rvjit

#endif
