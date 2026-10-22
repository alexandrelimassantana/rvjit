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

#include "rvjit/interface/stream/stream_table.hpp"

namespace rvjit {

namespace {

/// A stream that is only a slot of the table
struct slot_stream_t : stream::stream_impl_t {
    static slot_stream_t *invalid() {
        static slot_stream_t none(true);
        return &none;
    }

    explicit slot_stream_t(bool dead = false) : dead_(dead) {}

    bool is_valid() const override { return !dead_; }

private:
    bool dead_;
};

/// A factory handing out the slots of a three-slot table
struct slot_table_t : stream::stream_table_t<slot_stream_t, 3> {
    slot_stream_t *create() {
        slot_stream_t *s = next_();
        if (!s) return invalid();
        take_();
        return s;
    }
};

} // namespace

/// A dropped slot is handed out again, a committed one is kept
TEST(stream_table, drop_reclaims_slots_and_commit_keeps_them) {
    slot_table_t table;
    slot_stream_t *kept = table.create();

    ASSERT_TRUE(table.begin());
    slot_stream_t *dropped = table.create();
    EXPECT_NE(dropped, kept);
    ASSERT_TRUE(table.drop());
    EXPECT_EQ(table.create(), dropped);

    ASSERT_TRUE(table.begin());
    slot_stream_t *committed = table.create();
    ASSERT_TRUE(table.commit());
    EXPECT_NE(table.create(), committed);
}

/// A full table yields the invalid stream until a drop makes room
TEST(stream_table, a_full_table_yields_the_invalid_stream) {
    slot_table_t table;
    ASSERT_TRUE(table.begin());
    for (int i = 0; i < slot_table_t::CAPACITY; ++i)
        EXPECT_TRUE(table.create()->is_valid());
    EXPECT_FALSE(table.create()->is_valid());
    ASSERT_TRUE(table.drop());
    EXPECT_TRUE(table.create()->is_valid());
}

} // namespace rvjit
