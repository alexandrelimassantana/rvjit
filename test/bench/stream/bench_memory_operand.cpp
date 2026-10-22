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

// Example/probe binary for rvjit/interface/stream/memory/memory_operand.hpp's
// memory operand streams: registers one (an iterator paired with memop emission
// and a destination register ring) from the same caller-supplied
// parameters bench_iterator.cpp takes, and stores each loaded value out to
// a results buffer for host-side verification against a plain row-major
// address oracle -- see stream_common.hpp.

#include <cstdio>

#include "rvjit/implementation/rv64/isa/buffer/stack.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/const_arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/transfer/transfer.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "stream_common.hpp"

int main(int argc, char **argv) {
    using namespace rvjit_bench;

    const auto toks = expand_argv(argc, argv);
    stream_config_t cfg;
    if (!parse_stream_config(toks, cfg)) {
        print_stream_usage(argv[0]);
        return 1;
    }

    class memory_operand_probe_t : public Xbyak_riscv::CodeGenerator {
    public:
        explicit memory_operand_probe_t(const stream_config_t &cfg)
            : CodeGenerator(1 << 20), cfg_(cfg) {}

        bool create_kernel() {
            generate();
            ready();
            return ok_ && getCode() != nullptr;
        }

        using kernel_fn_t = void (*)(const void *, void *);
        void run(const void *data, void *results) const {
            auto *fptr = reinterpret_cast<kernel_fn_t>(
                    const_cast<uint8_t *>(getCode()));
            fptr(data, results);
        }

    private:
        stream_config_t cfg_;
        bool ok_ = true;

        void generate() {
            using namespace Xbyak_riscv;
            using namespace rvjit;

            emitter_t e(*this);
            rv64::const_arithmetic::const_arithmetic_t arith(e);

            const Reg data_ptr = a0, results_base = a1;
            rvv::buffer::architecture_t source
                    = rvv::buffer::architecture_t::default_list();
            source.integer.without({data_ptr, results_base});
            rvv::buffer::buffer_pool_t pool(source);
            rv64::transfer::transfer_t rt(e, arith);
            rv64::stack_t persist(e, pool);
            rv64::stream::iterator::iterator_t istream(e, arith, rt, pool);
            rvv::memory::memory_t mem(e);
            stream::memory_operand::memory_operand_t memops(
                    e, istream, mem, pool);

            plan_t req;
            req.origin = pointer_t(data_ptr);
            req.addressing = cfg_.addressing;
            req.update = cfg_.end;
            req.strategy = cfg_.strategy_hint;
            req.space
                    .axis(cfg_.shape[0],
                            stride_const(*this, pool, cfg_.stride[0]))
                    .axis(cfg_.shape[1],
                            stride_const(*this, pool, cfg_.stride[1]));

            // A 1-deep destination ring: this probe consumes every loaded
            // value immediately (ascending order, one extra store each),
            // so it never needs more than one register live at a time
            stream::memory_operand::spec_t spec;
            spec.op = memory::op_t().scalar().load().as_type(cfg_.dt);
            spec.origin = req.origin;
            spec.space = req.space;
            spec.addressing = cfg_.addressing;
            spec.update = cfg_.end;
            stream::memory_operand::variant_t variant;
            variant.strategy = cfg_.strategy_hint;
            variant.ntmp = cfg_.ntmp;
            variant.ring = rvjit::buffer_map_t({1, 1});
            stream::memory_operand::memory_operand_impl_t *mo
                    = memops.create(spec, variant);
            if (!mo->is_valid()) {
                ok_ = false;
                ret();
                return;
            }

            const name_t results_name = pool.integers().pop();
            if (results_name == no_name) {
                ok_ = false;
                ret();
                return;
            }
            const Reg results_ptr = Reg(results_name);

            persist.preserve();

            mv(results_ptr, results_base);
            const int width = sewb(cfg_.dt);
            const memory::op_t st
                    = memory::op_t().scalar().store().as_type(cfg_.dt);

            for (int i0 = 0; i0 < cfg_.shape[0]; ++i0)
                for (int i1 = 0; i1 < cfg_.shape[1]; ++i1)
                    mo->init_iteration({i0, i1});

            for (int i0 = 0; i0 < cfg_.shape[0]; ++i0) {
                for (int i1 = 0; i1 < cfg_.shape[1]; ++i1) {
                    const index_t idx {i0, i1};
                    mo->iteration(idx); // emits the load, then advances
                    memory::op_t store_op = st;
                    store_op.data.name = mo->get(idx);
                    store_op.addr = pointer_t(results_ptr);
                    mem.store(store_op);
                    addi(results_ptr, results_ptr, width);
                }
            }

            persist.restore();
            ret();
        }
    };

    memory_operand_probe_t kernel(cfg);
    return run_and_report(cfg, kernel, "bench_memory_operand");
}
