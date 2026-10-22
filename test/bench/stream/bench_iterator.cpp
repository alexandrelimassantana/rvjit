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

// Example/probe binary for rvjit/interface/stream/iterator/iterator.hpp's
// iterator streams: registers one from caller-supplied
// shape/stride/strategy/addressing/end parameters, manually emits a scalar
// load per position (an iterator stream has no memop -- see
// bench_memory_operand.cpp for the memop-owning counterpart),
// and stores each loaded value out to a results buffer for host-side
// verification against a plain row-major address oracle.

#include <cstdio>

#include "rvjit/implementation/rv64/isa/buffer/stack.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/const_arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/transfer/transfer.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "stream_common.hpp"

namespace rvjit_bench {

enum class run_mode_t { correctness, inspection };

void print_iterator_usage(const char *prog) {
    print_stream_usage(prog);
    std::fprintf(stderr,
            "  --mode=correctness|inspection: correctness (default) loads "
            "and stores each position for host-side verification; "
            "inspection emits only the iterator's own init/produce "
            "pointer-update instructions, skipping the load/store pair -- "
            "pair with --dump-asm to inspect addressing codegen without "
            "load/store noise.\n");
}

} // namespace rvjit_bench

int main(int argc, char **argv) {
    using namespace rvjit_bench;

    const auto toks = expand_argv(argc, argv);

    run_mode_t mode = run_mode_t::correctness;
    std::vector<std::string> rest;
    for (auto &tok : toks) {
        std::string name, value;
        if (!parse_flag(tok, name, value)) {
            rest.push_back(tok);
        } else if (name == "mode") {
            mode = value == "inspection" ? run_mode_t::inspection
                                         : run_mode_t::correctness;
        } else if (name == "help") {
            print_iterator_usage(argv[0]);
            return 0;
        } else {
            rest.push_back(tok);
        }
    }

    stream_config_t cfg;
    if (!parse_stream_config(rest, cfg)) {
        print_iterator_usage(argv[0]);
        return 1;
    }

    class iterator_probe_t : public Xbyak_riscv::CodeGenerator {
    public:
        explicit iterator_probe_t(const stream_config_t &cfg, run_mode_t mode)
            : CodeGenerator(1 << 20), cfg_(cfg), addressing_(mode) {}

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
        run_mode_t addressing_;
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

            auto plan = istream.plan(req, cfg_.ntmp);
            istream.new_iterator(plan);
            if (!plan.is_valid() || plan.needs_regs()) {
                ok_ = false;
                ret();
                return;
            }

            stream::iterator::iterator_impl_t *it = istream.iterator(plan);
            if (!it->is_valid()) {
                ok_ = false;
                ret();
                return;
            }

            const bool validate = addressing_ == run_mode_t::correctness;
            Reg results_ptr, scratch;
            if (validate) {
                const name_t results_name = pool.integers().pop();
                const name_t scratch_name = pool.integers().pop();
                if (results_name == no_name || scratch_name == no_name) {
                    ok_ = false;
                    ret();
                    return;
                }
                results_ptr = Reg(results_name);
                scratch = Reg(scratch_name);
            }

            persist.preserve();

            if (validate) mv(results_ptr, results_base);

            const int width = sewb(cfg_.dt);
            const memory::op_t ld
                    = memory::op_t().scalar().load().as_type(cfg_.dt);
            const memory::op_t st
                    = memory::op_t().scalar().store().as_type(cfg_.dt);

            // Init sweep first, ascending nested order: builds whatever
            // pivot chains this plan's axes need (a no-op for a plan that
            // resolved every axis to `direct`)
            for (int i0 = 0; i0 < cfg_.shape[0]; ++i0)
                for (int i1 = 0; i1 < cfg_.shape[1]; ++i1)
                    it->init_iteration({i0, i1});

            // Main sweep: read this position's address, load+store it
            // (skipped under --mode=inspection, leaving only the
            // iterator's own pointer-update instructions below), then
            // advance for the next position -- the same order
            // memory_operand_t::produce_() itself uses internally
            for (int i0 = 0; i0 < cfg_.shape[0]; ++i0) {
                for (int i1 = 0; i1 < cfg_.shape[1]; ++i1) {
                    const index_t idx {i0, i1};
                    if (validate) {
                        const pointer_t addr = it->pointer(idx);
                        memory::op_t load_op = ld;
                        load_op.data.name = reg_name(scratch);
                        load_op.addr = addr;
                        mem.load(load_op);
                        memory::op_t store_op = st;
                        store_op.data.name = reg_name(scratch);
                        store_op.addr = pointer_t(results_ptr);
                        mem.store(store_op);
                        addi(results_ptr, results_ptr, width);
                    }
                    it->iteration(idx);
                }
            }

            persist.restore();
            ret();
        }
    };

    iterator_probe_t kernel(cfg, mode);
    return run_and_report(
            cfg, kernel, "bench_iterator", mode == run_mode_t::correctness);
}
