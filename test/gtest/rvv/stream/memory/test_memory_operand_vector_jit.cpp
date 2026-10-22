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

#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvv/stream/memory/memory_operand_helpers.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using namespace rvjit::stream::iterator;

// JIT-executed: 1D vector load+store round trip via iteration()

/// Builds a probe that loads `rows` vector rows through one memory operand
/// and stores them through another, sharing one ring-buffer register
rv64_jit_probe_t::generate_fn_t vector_roundtrip_probe(int rows, bool strided) {
    return [rows, strided](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(vgroup_size(LMUL::m1));

        const Reg in_ptr = a0;
        const Reg out_ptr = a1;
        const Reg avl = a2;

        static constexpr int kAvl = 4;
        gen.li(avl, kAvl);
        gen.vsetvli(x0, avl, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

        const int row_stride = kAvl * static_cast<int>(sizeof(float));

        Reg stride_reg {};
        if (strided) {
            stride_reg = Reg(env.pool.integers().pop());
            gen.li(stride_reg, static_cast<int>(sizeof(float)));
        }

        space_t in_space, out_space;
        in_space.axis(rows, const_t(row_stride));
        out_space.axis(rows, const_t(row_stride));

        // Vector-kind memops require `base_only` addressing: the rvv vector
        // ops have no immediate-offset form
        memory::op_t load_op
                = memory::op_t().vector().load().as_type(type_t::f32);
        if (strided) load_op.strided(stride_reg);
        memory::op_t store_op = load_op;
        store_op.store();

        const name_t id = env.pool.vectors().pop();
        const buffer::buffer_t regs(&id, 1);
        const buffer_map_t ring(regs, {1, 1});

        auto *load_stream
                = env.memops.create(make_spec(load_op, pointer_t(in_ptr, 0),
                                            in_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        auto *store_stream
                = env.memops.create(make_spec(store_op, pointer_t(out_ptr, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        ASSERT_TRUE(load_stream->is_valid());
        ASSERT_TRUE(store_stream->is_valid());

        for (int r = 0; r < rows; ++r) {
            load_stream->iteration({r});
            store_stream->iteration({r});
        }
        gen.ret();
    };
}

class memory_operand_vector_roundtrip_t
    : public ::testing::TestWithParam<std::tuple<int, bool>> {};

/// A vector load/store round trip reproduces the input, unit-stride and strided
TEST_P(memory_operand_vector_roundtrip_t, values) {
    int rows;
    bool strided;
    std::tie(rows, strided) = GetParam();
    static constexpr int kAvl = 4;

    rv64_jit_probe_t kernel(vector_roundtrip_probe(rows, strided));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<float> in(rows * kAvl), out(rows * kAvl, 0.0f);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = static_cast<float>(i) + 0.25f;
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], in[i]) << "at i=" << i;
}

/// Unit-stride and strided addressing, single and multi-row sizes
INSTANTIATE_TEST_SUITE_P(AddressingModes, memory_operand_vector_roundtrip_t,
        ::testing::Values(std::make_tuple(3, false), std::make_tuple(3, true),
                std::make_tuple(1, false)));

} // namespace rvjit
