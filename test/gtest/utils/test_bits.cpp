#include "gtest/gtest.h"

#include "rvjit/utils/utils.hpp"

namespace rvjit {

TEST(bits, popcount_counts_set_bits) {
    EXPECT_EQ(popcount(0u), 0);
    EXPECT_EQ(popcount(1u), 1);
    EXPECT_EQ(popcount(0x80000000u), 1);
    EXPECT_EQ(popcount(0xf0f0f0f0u), 16);
    EXPECT_EQ(popcount(~0u), 32);
}

TEST(bits, ctz_finds_the_lowest_set_bit) {
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(ctz(1u << i), i);
        EXPECT_EQ(ctz(~0u << i), i);
    }
    EXPECT_EQ(ctz(0x80000001u), 0);
}

} // namespace rvjit
