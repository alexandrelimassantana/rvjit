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

#include "gtest/gtest.h"

#include "env.hpp"
#include "rv64_jit_test_generator.hpp"

namespace rvjit {

using namespace Xbyak_riscv;

/// Fills in `op`'s own `data`/`addr` for one position, then dispatches
/// through `dialect.load(op)`/`dialect.store(op)` per `op.dir`
static void emit(const operation::memory_t &dialect, memory::op_t op,
        name_t data, const pointer_t &addr) {
    op.data.name = data;
    op.addr = addr;
    op.dir == memory::dir_t::load ? dialect.load(op) : dialect.store(op);
}

/// Unsupported byte widths emit nothing
TEST(memory_move_emission, invalid_parameters) {
    // Invalid combinations: all emit nothing, leaving only ret().
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::memory::load_t load(env.e);
        rvv::memory::store_t store(env.e);

        // None of these should emit anything
        load.fload(ft0, t0, 1); // fload has no 1-byte case
        load.xload(t0, t0, 3, true); // xload has no 3-byte case
        store.fstore(ft0, t0, 1); // fstore has no 1-byte case
        store.xstore(t0, t0, 3); // xstore has no 3-byte case

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_EQ(kernel.getSize(), sizeof(uint32_t));
}

/// Every scalar width/signedness combination emits its load/store mnemonic
TEST(memory_move_emission, scalar_instructions) {
    static constexpr int n_pairs = 16;

    // Valid scalar combinations: probe and reference instructions interleaved
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        env_t env(gen);
        rvv::memory::load_t load(env.e);
        rvv::memory::store_t store(env.e);

        // Common instruction parameters
        const Reg ptr = t0;
        const int off = 0;

        // Float data types
        {
            const FReg data = ft0;

            for (auto sewb : {2, 4, 8}) {
                // Load
                load.fload(data, pointer_t(ptr, off), sewb);
                switch (sewb) {
                    case 2: gen.flh(data, ptr, off); break;
                    case 4: gen.flw(data, ptr, off); break;
                    case 8: gen.fld(data, ptr, off); break;
                }
                // Store
                store.fstore(data, pointer_t(ptr, off), sewb);
                switch (sewb) {
                    case 2: gen.fsh(data, ptr, off); break;
                    case 4: gen.fsw(data, ptr, off); break;
                    case 8: gen.fsd(data, ptr, off); break;
                }
            }
        }

        // Integer loads
        {
            const Reg data = t1;

            // Signed
            for (auto sewb : {1, 2, 4, 8}) {
                // Load
                load.xload(data, pointer_t(ptr, off), sewb, true);
                switch (sewb) {
                    case 1: gen.lb(data, ptr, off); break;
                    case 2: gen.lh(data, ptr, off); break;
                    case 4: gen.lw(data, ptr, off); break;
                    case 8: gen.ld(data, ptr, off); break;
                }
                // Store
                store.xstore(data, pointer_t(ptr, off), sewb);
                switch (sewb) {
                    case 1: gen.sb(data, ptr, off); break;
                    case 2: gen.sh(data, ptr, off); break;
                    case 4: gen.sw(data, ptr, off); break;
                    case 8: gen.sd(data, ptr, off); break;
                }
            }

            // Unsigned
            load.xload(data, pointer_t(ptr, off), 1, false);
            gen.lbu(data, ptr, off);
            load.xload(data, pointer_t(ptr, off), 2, false);
            gen.lhu(data, ptr, off);
        }

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// Every vector dtype/addressing combination emits its load/store mnemonic
TEST(memory_move_emission, vector_instructions) {
    using op_t = type_t;
    static constexpr int n_pairs = 18;

    // Valid vector memory operation parameter combinations
    rv64_jit_probe_t kernel([](rv64_jit_test_generator_t &gen) {
        model_env_t env(gen);

        // Common instruction parameters
        const VReg data = VReg(0);
        const Reg ptr = t0;
        const Reg stride = t1;
        const VReg index = VReg(1);

        for (auto dt : {op_t::u8, op_t::f16, op_t::f64}) {

            // Unit-stride
            auto desc = memory::op_t().vector().as_type(dt);

            // Load
            emit(env.mem, desc.load(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vle8_v(data, ptr); break;
                case op_t::f16: gen.vle16_v(data, ptr); break;
                case op_t::f64: gen.vle64_v(data, ptr); break;
                default: break;
            }

            // Store
            emit(env.mem, desc.store(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vse8_v(data, ptr); break;
                case op_t::f16: gen.vse16_v(data, ptr); break;
                case op_t::f64: gen.vse64_v(data, ptr); break;
                default: break;
            }

            // Constant-stride
            desc.strided(stride);

            // Load
            emit(env.mem, desc.load(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vlse8_v(data, ptr, stride); break;
                case op_t::f16: gen.vlse16_v(data, ptr, stride); break;
                case op_t::f64: gen.vlse64_v(data, ptr, stride); break;
                default: break;
            }

            // Store
            emit(env.mem, desc.store(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vsse8_v(data, ptr, stride); break;
                case op_t::f16: gen.vsse16_v(data, ptr, stride); break;
                case op_t::f64: gen.vsse64_v(data, ptr, stride); break;
                default: break;
            }

            // Indexed
            desc.indexed(index);

            // Load
            emit(env.mem, desc.load(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vluxei8_v(data, ptr, index); break;
                case op_t::f16: gen.vluxei16_v(data, ptr, index); break;
                case op_t::f64: gen.vluxei64_v(data, ptr, index); break;
                default: break;
            }

            // Store
            emit(env.mem, desc.store(), 0, ptr);
            switch (dt) {
                case op_t::u8: gen.vsuxei8_v(data, ptr, index); break;
                case op_t::f16: gen.vsuxei16_v(data, ptr, index); break;
                case op_t::f64: gen.vsuxei64_v(data, ptr, index); break;
                default: break;
            }
        }

        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    expect_matching_pairs(kernel, n_pairs);
}

/// `supports_*` validates widths and forms without register names
TEST(memory_supports, widths) {
    Xbyak_riscv::CodeGenerator gen;
    model_env_t env(gen);
    using memory::op_t;

    {
        const auto op = op_t().scalar().load().as_type(type_t::f32);
        EXPECT_TRUE(env.mem.supports_load(op));
    }
    {
        const auto op = op_t().scalar().store().as_type(type_t::s8);
        EXPECT_TRUE(env.mem.supports_store(op));
    }
    {
        const auto op = op_t().vector().load().as_type(type_t::f32);
        EXPECT_TRUE(env.mem.supports_load(op));
    }
}

} // namespace rvjit
