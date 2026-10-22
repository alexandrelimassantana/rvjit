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

#include "rvjit/interface/linalg/matmul/strides.hpp"

namespace rvjit {

using linalg::matmul::granule_stride_t;
using linalg::matmul::strides_t;
using base_t = granule_stride_t::base_t;

namespace {

linalg::matmul::plan_t make_plan(type_t dta, type_t dtb, type_t dtc,
        bool transa = false, bool transb = false) {
    linalg::matmul::plan_t p;
    p.dta = dta;
    p.dtb = dtb;
    p.dtc = dtc;
    p.is_transa = transa;
    p.is_transb = transb;
    return p;
}

} // namespace

/// Unit grains leave the strides of one element step
TEST(matmul_strides, unit_grains) {
    const auto p = make_plan(type_t::f32, type_t::f32, type_t::f32);
    const strides_t s = strides_t::of(p, 1, 1, 1);
    EXPECT_EQ(s.a_k, (granule_stride_t {base_t::lda, 1}));
    EXPECT_EQ(s.a_m, (granule_stride_t {base_t::bytes, 4}));
    EXPECT_EQ(s.b_k, (granule_stride_t {base_t::bytes, 4}));
    EXPECT_EQ(s.b_n, (granule_stride_t {base_t::ldb, 1}));
    EXPECT_EQ(s.c_n, (granule_stride_t {base_t::ldc, 1}));
    EXPECT_EQ(s.c_m, (granule_stride_t {base_t::bytes, 4}));
}

/// The grain of each axis scales the stride along that axis, and no other
TEST(matmul_strides, grains_scale_their_own_axis) {
    const auto p = make_plan(type_t::f32, type_t::f32, type_t::f32);
    const strides_t s = strides_t::of(p, 16, 4, 2);
    EXPECT_EQ(s.a_k, (granule_stride_t {base_t::lda, 2}));
    EXPECT_EQ(s.a_m, (granule_stride_t {base_t::bytes, 64}));
    EXPECT_EQ(s.b_k, (granule_stride_t {base_t::bytes, 8}));
    EXPECT_EQ(s.b_n, (granule_stride_t {base_t::ldb, 4}));
    EXPECT_EQ(s.c_n, (granule_stride_t {base_t::ldc, 4}));
    EXPECT_EQ(s.c_m, (granule_stride_t {base_t::bytes, 64}));
}

/// A transposed operand swaps which axis jumps a leading dimension
TEST(matmul_strides, transposition_swaps_the_axes) {
    const auto p = make_plan(type_t::f32, type_t::f32, type_t::f32, true, true);
    const strides_t s = strides_t::of(p, 16, 4, 2);
    EXPECT_EQ(s.a_k, (granule_stride_t {base_t::bytes, 8}));
    EXPECT_EQ(s.a_m, (granule_stride_t {base_t::lda, 16}));
    EXPECT_EQ(s.b_k, (granule_stride_t {base_t::ldb, 2}));
    EXPECT_EQ(s.b_n, (granule_stride_t {base_t::bytes, 16}));
}

/// Each operand's element size sets its own byte strides
TEST(matmul_strides, element_sizes_follow_the_data_types) {
    const auto p = make_plan(type_t::s8, type_t::u8, type_t::s32);
    const strides_t s = strides_t::of(p, 8, 2, 4);
    EXPECT_EQ(s.a_m, (granule_stride_t {base_t::bytes, 8}));
    EXPECT_EQ(s.b_k, (granule_stride_t {base_t::bytes, 4}));
    EXPECT_EQ(s.c_m, (granule_stride_t {base_t::bytes, 32}));

    const auto t = make_plan(type_t::s8, type_t::u8, type_t::s32, true, true);
    const strides_t st = strides_t::of(t, 8, 2, 4);
    EXPECT_EQ(st.a_k, (granule_stride_t {base_t::bytes, 4}));
    EXPECT_EQ(st.b_n, (granule_stride_t {base_t::bytes, 2}));
}

/// An extended B keeps the element size of its memory type
TEST(matmul_strides, an_extended_b_keeps_its_memory_element_size) {
    auto p = make_plan(type_t::s32, type_t::s8, type_t::s32);
    p.dtb.reg = type_t::s32;
    const strides_t s = strides_t::of(p, 16, 4, 2);
    EXPECT_EQ(s.b_k, (granule_stride_t {base_t::bytes, 2}));
    EXPECT_EQ(s.a_m, (granule_stride_t {base_t::bytes, 64}));
}

} // namespace rvjit
