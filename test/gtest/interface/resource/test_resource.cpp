#include "gtest/gtest.h"

#include "rvjit/interface/resource/resource.hpp"

namespace rvjit {

TEST(demand_t, index_inverts_at_for_every_class) {
    for (int i = 0; i < demand_t::CLASSES; ++i)
        EXPECT_EQ(demand_t::index(demand_t::at(i)), i);
}

TEST(demand_t, unknowns_marks_every_class) {
    const demand_t d = demand_t::unknowns();
    for (int i = 0; i < demand_t::CLASSES; ++i)
        EXPECT_FALSE(d.knows(demand_t::at(i)));
    EXPECT_EQ(d.unknown, static_cast<unsigned>(demand_t::ALL));
}

} // namespace rvjit
