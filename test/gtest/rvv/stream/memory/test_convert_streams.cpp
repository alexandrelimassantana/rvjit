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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/arithmetic.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/stream/memory/input_stream.hpp"
#include "rvjit/interface/stream/memory/output_stream.hpp"
#include "rvv/stream/memory/memory_operand_helpers.hpp"

namespace rvjit {

using namespace Xbyak_riscv;
using rvjit::stream::policy_t;
using rvjit::stream::memory_operand::input_stream_t;
using rvjit::stream::memory_operand::output_stream_t;

namespace {

constexpr int kAvl = 4; // Elements of a row

// Values exactly representable in f16, and their bit patterns
constexpr float kValues[8]
        = {1.0f, 2.0f, 0.5f, -1.5f, 3.0f, 0.25f, -0.75f, 8.0f};
constexpr uint16_t kHalf[8]
        = {0x3c00, 0x4000, 0x3800, 0xbe00, 0x4200, 0x3400, 0xba00, 0x4800};

/// A probe loading `rows` f32 rows into wide registers (groups of two),
/// then narrowing and storing them to f16 through an output stream
rv64_jit_probe_t::generate_fn_t narrow_store_probe(int rows, int delay) {
    return [rows, delay](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(2);
        gen.li(a2, kAvl);
        gen.vsetvli(x0, a2, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);

        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        const buffer_map_t ring(regs, {rows, 1});

        space_t in_space, out_space;
        in_space.axis(rows, const_t(kAvl * 4));
        out_space.axis(rows, const_t(kAvl * 2));
        memory::op_t load_op
                = memory::op_t().vector().load().as_type(type_t::f32);
        memory::op_t store_op
                = memory::op_t().vector().store().as_type(type_t::f16);

        auto *load
                = env.memops.create(make_spec(load_op, pointer_t(a0, 0),
                                            in_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        auto *store
                = env.memops.create(make_spec(store_op, pointer_t(a1, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        ASSERT_TRUE(load->is_valid());
        ASSERT_TRUE(store->is_valid());
        load->emit();

        output_stream_t out;
        {
            const bool ok = out.init(env.pm.isa().arithmetic(),
                    env.pm.streams().scheduler(), store,
                    type_parameter_t(type_t::f16, type_t::f32),
                    policy_t::literal(delay));
            ASSERT_TRUE(ok);
        }
        gen.vsetvli(x0, x0, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        out.emit();
        gen.ret();
    };
}

class narrow_store_t : public ::testing::TestWithParam<std::tuple<int, int>> {};

/// The accumulators are narrowed in place and stored, for any delay
TEST_P(narrow_store_t, values) {
    int rows, delay;
    std::tie(rows, delay) = GetParam();
    rv64_jit_probe_t kernel(narrow_store_probe(rows, delay));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<float> in(rows * kAvl);
    std::vector<uint16_t> out(rows * kAvl, 0);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = kValues[i % 8];
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], kHalf[i % 8]) << "at i=" << i;
}

INSTANTIATE_TEST_SUITE_P(Delays, narrow_store_t,
        ::testing::Values(std::make_tuple(1, 0), std::make_tuple(4, 0),
                std::make_tuple(4, 2), std::make_tuple(4, 100),
                std::make_tuple(6, 3)));

/// The narrowings (N) and stores (S) of the code an output stream emits,
/// in order
std::string narrow_store_order(int rows, int delay, type_t acc) {
    rv64_jit_probe_t kernel([=](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(2);
        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        const buffer_map_t ring(regs, {rows, 1});
        space_t out_space;
        out_space.axis(rows, const_t(kAvl * 2));
        memory::op_t store_op
                = memory::op_t().vector().store().as_type(type_t::f16);
        auto *store
                = env.memops.create(make_spec(store_op, pointer_t(a1, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        output_stream_t out;
        {
            const bool ok = out.init(env.pm.isa().arithmetic(),
                    env.pm.streams().scheduler(), store,
                    type_parameter_t(type_t::f16, acc),
                    policy_t::literal(delay));
            ASSERT_TRUE(ok);
        }
        out.emit();
        gen.ret();
    });
    EXPECT_TRUE(kernel.create_kernel());

    std::string order;
    const auto *code = kernel.getCode<const uint32_t *>();
    for (size_t i = 0; i < kernel.getSize() / sizeof(uint32_t); ++i) {
        const uint32_t w = code[i];
        if ((w & 0xfe0ff07fu) == 0x4a0a1057u) order += 'N'; // vfncvt.f.f.w
        if ((w & 0x0000707fu) == 0x00005027u) order += 'S'; // vse16.v
    }
    return order;
}

/// Narrowings run `delay` positions ahead of the stores, clamped to the count
TEST(output_stream_order, narrowings_lead_the_stores) {
    EXPECT_EQ(narrow_store_order(4, 0, type_t::f32), "NSNSNSNS");
    EXPECT_EQ(narrow_store_order(4, 1, type_t::f32), "NNSNSNSS");
    EXPECT_EQ(narrow_store_order(4, 2, type_t::f32), "NNNSNSSS");
    EXPECT_EQ(narrow_store_order(4, 100, type_t::f32), "NNNNSSSS");
}

/// An output that is not narrower is only stored
TEST(output_stream_order, equal_types_only_store) {
    EXPECT_EQ(narrow_store_order(4, 2, type_t::f16), "SSSS");
}

/// A pair of types that does not narrow by half is rejected
TEST(output_stream_init, rejects_other_ratios) {
    host_model_env_t env;
    env.pool.set_vector_group_size(2);
    const buffer::buffer_t regs = env.pool.vectors().pop(1);
    space_t space;
    space.axis(1, const_t(kAvl * 2));
    memory::op_t store_op
            = memory::op_t().vector().store().as_type(type_t::f16);
    auto *store = env.memops.create(make_spec(store_op, pointer_t(a1, 0), space,
                                            addressing_t::base_only),
            make_variant(rv64::stream::iterator::guided, 0,
                    buffer_map_t(regs, {1, 1})));
    ASSERT_TRUE(store->is_valid());
    output_stream_t out;
    {
        const bool ok = out.init(env.pm.isa().arithmetic(),
                env.pm.streams().scheduler(), store,
                type_parameter_t(type_t::f16, type_t::f64), policy_t::zero());
        EXPECT_FALSE(ok);
    }
    {
        const bool ok = out.init(env.pm.isa().arithmetic(),
                env.pm.streams().scheduler(), store,
                type_parameter_t(type_t::f16, type_t::s32), policy_t::zero());
        EXPECT_FALSE(ok);
    }
}

/// A configuration switch that fails is reported and retried, never assumed
TEST(conversion_stream, a_failed_switch_keeps_the_width) {
    host_model_env_t env;
    env.pool.set_vector_group_size(2);
    const buffer::buffer_t regs = env.pool.vectors().pop(1);
    space_t space;
    space.axis(1, const_t(kAvl * 2));
    memory::op_t load_op = memory::op_t().vector().load().as_type(type_t::f16);
    auto *load = env.memops.create(make_spec(load_op, pointer_t(a0, 0), space,
                                           addressing_t::base_only),
            make_variant(rv64::stream::iterator::guided, 0,
                    buffer_map_t(regs, {1, 1})));
    ASSERT_TRUE(load->is_valid());
    conversion_t conv;
    ASSERT_TRUE(conv.plan(type_t::f16, type_t::f32, form_t::vector,
            env.pm.isa().arithmetic()));

    int calls = 0;
    const auto refuse = [&calls](type_t) { return ++calls < 0; };
    stream::memory_operand::conversion_stream_t s;
    ASSERT_TRUE(s.init(
            env.pm.isa().arithmetic(), conv, load, type_t::f32, 1, refuse));
    EXPECT_FALSE(s.configure(0));
    EXPECT_FALSE(s.configure(0));
    EXPECT_EQ(calls, 2);
    EXPECT_TRUE(s.restore());
    EXPECT_EQ(calls, 2);
}

/// A probe loading f16 rows into the upper half of wide registers, widening
/// them there, and storing the f32 results
rv64_jit_probe_t::generate_fn_t widen_load_probe(int rows) {
    return [rows](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(2);
        gen.li(a2, kAvl);
        gen.vsetvli(x0, a2, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);

        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        const buffer_map_t wide(regs, {rows, 1});
        const int offset = 1;
        memory_variant_t upper
                = make_variant(rv64::stream::iterator::guided, 0, wide);
        upper.offset = offset;

        space_t in_space, out_space;
        in_space.axis(rows, const_t(kAvl * 2));
        out_space.axis(rows, const_t(kAvl * 4));
        memory::op_t load_op
                = memory::op_t().vector().load().as_type(type_t::f16);
        memory::op_t store_op
                = memory::op_t().vector().store().as_type(type_t::f32);

        auto *load
                = env.memops.create(make_spec(load_op, pointer_t(a0, 0),
                                            in_space, addressing_t::base_only),
                        upper);
        auto *store
                = env.memops.create(make_spec(store_op, pointer_t(a1, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, wide));
        ASSERT_TRUE(load->is_valid());
        ASSERT_TRUE(store->is_valid());

        input_stream_t reload;
        {
            const bool ok = reload.init(env.pm.isa().arithmetic(), load,
                    type_parameter_t(type_t::f16, type_t::f32), offset);
            ASSERT_TRUE(ok);
        }
        ASSERT_TRUE(reload.converts());
        reload.emit();
        gen.vsetvli(x0, x0, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        store->emit();
        gen.ret();
    };
}

/// The narrow rows are widened into whole wide registers
TEST(input_stream, widens_from_the_upper_half) {
    static constexpr int rows = 3;
    rv64_jit_probe_t kernel(widen_load_probe(rows));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<uint16_t> in(rows * kAvl);
    std::vector<float> out(rows * kAvl, 0.0f);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = kHalf[i % 8];
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], kValues[i % 8]) << "at i=" << i;
}

/// Switches to `t`'s width from e32/m4, keeping the vector length
bool configure_from_e32_m4(rv64_jit_test_generator_t &gen, type_t t) {
    const int w = sewb(t);
    const SEW sew = w == 1 ? SEW::e8 : w == 2 ? SEW::e16 : SEW::e32;
    const LMUL lmul = w == 1 ? LMUL::m1 : w == 2 ? LMUL::m2 : LMUL::m4;
    gen.vsetvli(x0, x0, sew, lmul, VTA::ta, VMA::ma);
    return true;
}

// f32 values around the rounding and saturation margins of s8 and u8
constexpr float kWide[8]
        = {1.4f, 2.5f, -3.5f, 127.6f, -200.f, 300.f, 0.5f, -0.5f};
constexpr int8_t kS8[8] = {1, 2, -4, 127, -128, 127, 0, 0};
constexpr uint8_t kU8[8] = {1, 2, 0, 128, 0, 255, 0, 0};

/// A probe loading `rows` f32 rows into groups of four and storing them as
/// `out` through a multi-step output stream
rv64_jit_probe_t::generate_fn_t chain_store_probe(int rows, type_t out) {
    return [rows, out](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(4);
        gen.li(a2, kAvl);
        gen.vsetvli(x0, a2, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        const buffer_map_t ring(regs, {rows, 1});
        space_t in_space, out_space;
        in_space.axis(rows, const_t(kAvl * 4));
        out_space.axis(rows, const_t(kAvl));
        memory::op_t load_op
                = memory::op_t().vector().load().as_type(type_t::f32);
        memory::op_t store_op = memory::op_t().vector().store().as_type(out);
        auto *load
                = env.memops.create(make_spec(load_op, pointer_t(a0, 0),
                                            in_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        auto *store
                = env.memops.create(make_spec(store_op, pointer_t(a1, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, ring));
        ASSERT_TRUE(load->is_valid());
        ASSERT_TRUE(store->is_valid());
        load->emit();

        output_stream_t stream;
        const auto configure
                = [&gen](type_t t) { return configure_from_e32_m4(gen, t); };
        {
            const bool ok = stream.init(env.pm.isa().arithmetic(),
                    env.pm.streams().scheduler(), store,
                    type_parameter_t(out, type_t::f32), policy_t::literal(1),
                    configure);
            ASSERT_TRUE(ok);
        }
        stream.emit();
        gen.ret();
    };
}

/// f32 accumulators reach s8 through a conversion and two saturating
/// narrowings, rounding to nearest even
TEST(output_stream_chain, f32_to_s8) {
    static constexpr int rows = 2;
    rv64_jit_probe_t kernel(chain_store_probe(rows, type_t::s8));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<float> in(rows * kAvl);
    std::vector<int8_t> out(rows * kAvl, 99);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = kWide[i % 8];
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], kS8[i % 8]) << "at i=" << i;
}

/// f32 accumulators reach u8 through an unsigned conversion, which clamps the
/// negatives to zero, and two saturating narrowings
TEST(output_stream_chain, f32_to_u8) {
    static constexpr int rows = 2;
    rv64_jit_probe_t kernel(chain_store_probe(rows, type_t::u8));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<float> in(rows * kAvl);
    std::vector<uint8_t> out(rows * kAvl, 99);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = kWide[i % 8];
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], kU8[i % 8]) << "at i=" << i;
}

/// A probe loading `rows` rows of `in` into the upper quarter of groups of
/// four, extending and converting them to f32 there, and storing the f32
rv64_jit_probe_t::generate_fn_t chain_load_probe(int rows, type_t in) {
    return [rows, in](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(4);
        gen.li(a2, kAvl);
        gen.vsetvli(x0, a2, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);

        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        const buffer_map_t wide(regs, {rows, 1});
        const int offset = 3;
        memory_variant_t upper
                = make_variant(rv64::stream::iterator::guided, 0, wide);
        upper.offset = offset;
        space_t in_space, out_space;
        in_space.axis(rows, const_t(kAvl));
        out_space.axis(rows, const_t(kAvl * 4));
        memory::op_t load_op = memory::op_t().vector().load().as_type(in);
        memory::op_t store_op
                = memory::op_t().vector().store().as_type(type_t::f32);
        auto *load
                = env.memops.create(make_spec(load_op, pointer_t(a0, 0),
                                            in_space, addressing_t::base_only),
                        upper);
        auto *store
                = env.memops.create(make_spec(store_op, pointer_t(a1, 0),
                                            out_space, addressing_t::base_only),
                        make_variant(rv64::stream::iterator::guided, 0, wide));
        ASSERT_TRUE(load->is_valid());
        ASSERT_TRUE(store->is_valid());

        input_stream_t reload;
        const auto configure
                = [&gen](type_t t) { return configure_from_e32_m4(gen, t); };
        {
            const bool ok = reload.init(env.pm.isa().arithmetic(), load,
                    type_parameter_t(in, type_t::f32), offset, configure);
            ASSERT_TRUE(ok);
        }
        reload.fetch_convert([](int j) { return index_t {j, 0}; }, rows);
        store->emit();
        gen.ret();
    };
}

/// s8 values are sign-extended 4x into their f32 groups and converted there
TEST(input_stream_chain, s8_to_f32) {
    static constexpr int rows = 3;
    rv64_jit_probe_t kernel(chain_load_probe(rows, type_t::s8));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<int8_t> in(rows * kAvl);
    std::vector<float> out(rows * kAvl, 0.f);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = kS8[i % 8];
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], static_cast<float>(kS8[i % 8])) << "at i=" << i;
}

/// u8 values are zero-extended 4x and converted as unsigned
TEST(input_stream_chain, u8_to_f32) {
    static constexpr int rows = 3;
    rv64_jit_probe_t kernel(chain_load_probe(rows, type_t::u8));
    ASSERT_TRUE(kernel.create_kernel());

    std::vector<uint8_t> in(rows * kAvl);
    std::vector<float> out(rows * kAvl, 0.f);
    for (int i = 0; i < rows * kAvl; ++i)
        in[i] = static_cast<uint8_t>(200 + i);
    kernel(in.data(), out.data());
    for (int i = 0; i < rows * kAvl; ++i)
        ASSERT_EQ(out[i], static_cast<float>(200 + i)) << "at i=" << i;
}

/// The `vsetvli`s an f16 input stream emits over `rows` positions, converted
/// one by one through `iteration()` or in one `fetch_convert()` batch
int widen_switches(int rows, bool batched) {
    rv64_jit_probe_t kernel([=](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen, excluding_ints({a0, a1, a2}));
        env.pool.set_vector_group_size(4);
        const buffer::buffer_t regs = env.pool.vectors().pop(rows);
        memory_variant_t upper = make_variant(rv64::stream::iterator::guided, 0,
                buffer_map_t(regs, {rows, 1}));
        upper.offset = 2;
        space_t space;
        space.axis(rows, const_t(kAvl * 2));
        memory::op_t load_op
                = memory::op_t().vector().load().as_type(type_t::f16);
        auto *load = env.memops.create(make_spec(load_op, pointer_t(a0, 0),
                                               space, addressing_t::base_only),
                upper);
        input_stream_t in;
        const auto configure
                = [&gen](type_t t) { return configure_from_e32_m4(gen, t); };
        {
            const bool ok = in.init(env.pm.isa().arithmetic(), load,
                    type_parameter_t(type_t::f16, type_t::f32), 2, configure);
            ASSERT_TRUE(ok);
        }
        if (batched)
            in.fetch_convert([](int j) { return index_t {j, 0}; }, rows);
        else
            for (int j = 0; j < rows; ++j)
                in.iteration({j, 0});
        gen.ret();
    });
    EXPECT_TRUE(kernel.create_kernel());

    int n = 0;
    const auto *code = kernel.getCode<const uint32_t *>();
    for (size_t i = 0; i < kernel.getSize() / sizeof(uint32_t); ++i)
        if ((code[i] & 0x8000707fu) == 0x00007057u) ++n; // vsetvli
    return n;
}

/// A widening switches to the narrow width and back around every position
/// it converts alone, and once around a batch
TEST(input_stream_cost, widenings_switch_per_position_or_per_batch) {
    EXPECT_EQ(widen_switches(3, false), 6);
    EXPECT_EQ(widen_switches(3, true), 2);
}

} // namespace
} // namespace rvjit
