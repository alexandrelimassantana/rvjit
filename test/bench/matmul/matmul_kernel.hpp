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

#ifndef RVJIT_TEST_BENCH_MATMUL_KERNEL_HPP
#define RVJIT_TEST_BENCH_MATMUL_KERNEL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "matmul_case.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/rvjit.hpp"

namespace rvjit_bench {

using namespace rvjit;
using namespace rvjit::rvv;
using namespace rvjit::linalg::matmul;

/// The kernel's argument structure: pointers, and the runtime sizes/strides
/// (bytes for strides) a case sources from memory instead of a literal
struct matmul_args_t {
    const uint8_t *a;
    const uint8_t *b;
    uint8_t *c;
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    uint64_t alpha_bits, beta_bits; // The scalars' bit patterns
    const uint8_t *bias;
    int64_t rt_flags; // The runtime flag word
};

static_assert(int(post_alpha) == int(with_alpha)
                && int(post_beta) == int(with_beta)
                && int(post_bias) == int(with_bias)
                && int(post_postop) == int(with_postops),
        "the case's post flags mirror the matmul's codegen flags");

/// A native post-op multiplying the accumulators by 4, to exercise the chain
///
/// @details It computes through a scratch register: `s = a + a; a = s + s`
struct quadruple_postop_t : public postop_t {
    needs_t needs(const postop_shape_t &) const override {
        needs_t n;
        n.accs = 1;
        return n;
    }

    bool emit(postop_tile_t &t) override {
        auto op = rvjit::arithmetic::op_t::uniform(
                t.env.plan.epi_type(), t.env.c_form);
        t.for_each_acc([&](name_t a, const index_t &) {
            const name_t s = t.env.scratch.acc(0);
            op.rd.name = s;
            op.rs1.name = op.rs2.name = a;
            t.env.pm.isa().arithmetic().add(op);
            op.rd.name = a;
            op.rs1.name = op.rs2.name = s;
            t.env.pm.isa().arithmetic().add(op);
        });
        return true;
    }
};

/// JIT-compiles a matmul kernel specialized to one matmul_case_t
///
/// @note A/B/C layouts follow rvv::linalg::matmul::matmul_t's native stride
///     convention: A is `M x K` (column-major, `lda=M`) unless `transA`
///     (row-major, `lda=K`); B is `N x K` row-major (`ldb=K`) unless
///     `transB` (`K x N` row-major, `ldb=N`); C is always column-major
///     `M x N` (`ldc=M`)
class matmul_kernel_t : public Xbyak_riscv::CodeGenerator {
public:
    matmul_kernel_t(const matmul_case_t &c, const vpu_t &model)
        : Xbyak_riscv::CodeGenerator(1 << 20), c_(c), model_(model) {}

    /// Whether the kernel was generated; an assembler error (e.g. a branch
    /// out of range) is reported through `error()` instead of propagating
    bool create_kernel() {
        try {
            generate();
            ready();
        } catch (const Xbyak_riscv::Error &err) {
            error_ = Xbyak_riscv::ConvertErrorToString(err);
            return false;
        }
        return ok_ && getCode() != nullptr;
    }

    const std::string &error() const { return error_; }

    using kernel_fn_t = void (*)(const matmul_args_t *);

    void run(const matmul_args_t &args) const {
        auto *fptr = reinterpret_cast<kernel_fn_t>(
                const_cast<uint8_t *>(getCode()));
        fptr(&args);
    }

private:
    matmul_case_t c_;
    vpu_t model_;
    bool ok_ = true;
    std::string error_;

    /// A parameter loaded from the args structure at byte offset `off`
    static parameter_t from_args(size_t off) {
        return parameter_t(
                source_t(pointer_t(Xbyak_riscv::a0, static_cast<int>(off))));
    }

    void generate() {
        using namespace Xbyak_riscv;

        emitter_t e(*this);
        rv64::const_arithmetic::const_arithmetic_t arith(e);

        // a0 carries the args structure for the kernel's whole lifetime
        const Reg args = a0;
        rvv::buffer::architecture_t source
                = rvv::buffer::architecture_t::default_list();
        source.integer.without({a0, a1, a2});

        // Register-pressure stress cases cap the int candidate list from the
        // tail (s11 down to s0, then a7 down to a3, then t6 down to t0) to
        // force linalg::matmul::matmul_t::plan()'s tiered shed/direct-
        // fallback allocation into a specific rung on purpose
        if (c_.max_int_regs > 0) {
            const Reg all[] = {t0, t1, t2, t3, t4, t5, t6, a3, a4, a5, a6, a7,
                    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
            const int n_all = static_cast<int>(sizeof(all) / sizeof(all[0]));
            std::vector<Reg> excl;
            for (int i = c_.max_int_regs; i < n_all; ++i)
                excl.push_back(all[i]);
            source.integer.without(rvjit::buffer::buffer_t(
                    excl.data(), excl.data() + excl.size()));
        }

        rvv::buffer::buffer_pool_t pool(source);
        rv64::stack_t persist(e, pool);
        rv64::transfer::transfer_t rt(e, arith);
        rv64::control_flow::control_flow_t cf(e, arith, rt);
        rv64::loop::loopgen_t loops(e, arith, cf, pool, persist, rt);
        rv64::stream::iterator::iterator_t iterator_stream(e, arith, rt, pool);
        rvv::memory::memory_t mem_dialect(e);
        stream::memory_operand::memory_operand_t memops(
                e, iterator_stream, mem_dialect, pool);
        rvjit::rvv::programming_model_t pm(
                e, pool, loops, persist, rt, arith, cf, memops, model_);
        rvv::linalg::matmul::impl_t impl(pm);
        rvjit::linalg::matmul::matmul_t backend(impl);

        const int64_t lda_bytes = c_.lda_elems() * dtype_bytes(c_.dta);
        const int64_t ldb_bytes = c_.ldb_elems() * dtype_bytes(c_.dtb);
        const int64_t ldc_bytes = c_.ldc_elems() * dtype_bytes(c_.dtc);

        // The backend groups the vector registers itself, honoring a forced LMUL
        if (c_.lmul > 0) {
            const LMUL fixed = c_.lmul == 1 ? LMUL::m1
                    : c_.lmul == 2          ? LMUL::m2
                    : c_.lmul == 4          ? LMUL::m4
                                            : LMUL::m8;
            impl.set_lmul(fixed);
        }

        quadruple_postop_t quadruple;
        plan_t plan;
        plan.dta = type_parameter_t(c_.dta, c_.areg());
        plan.dtepi = c_.dtepi;
        plan.dtbias = c_.dtbias;
        plan.bias_kind
                = c_.bias_scalar ? bias_kind_t::scalar : bias_kind_t::per_m;
        plan.dtb = type_parameter_t(
                c_.dtb, c_.dtbreg == type_t::none ? c_.dtb : c_.dtbreg);
        plan.dtc = type_parameter_t(c_.dtc, c_.dtacc);
        plan.is_transa = c_.trans_a;
        plan.is_transb = c_.trans_b;

        // The epilogue: scalars are literal values or bit patterns loaded from
        // the args structure; the flag word follows the case's `rt`
        plan.flags = c_.post;
        const bool lit = c_.scalar_src == source_kind_t::literal;
        if (c_.post & post_alpha)
            plan.alpha = lit ? scalar_parameter_t(c_.alpha)
                             : scalar_parameter_t(from_args(
                                       offsetof(matmul_args_t, alpha_bits))
                                                          .source());
        if (c_.post & post_beta)
            plan.beta = lit ? scalar_parameter_t(c_.beta)
                            : scalar_parameter_t(from_args(
                                      offsetof(matmul_args_t, beta_bits))
                                                         .source());
        if (c_.post & post_bias)
            plan.bias = from_args(offsetof(matmul_args_t, bias));
        if (c_.post & post_postop) plan.add_postop(&quadruple);
        if (c_.rt == rt_kind_t::literal)
            plan.rt_flags = parameter_t(
                    c_.rt_mask == -1 ? c_.post : (c_.rt_mask & c_.post));
        else if (c_.rt == rt_kind_t::runtime)
            plan.rt_flags = from_args(offsetof(matmul_args_t, rt_flags));

        // Pointers always come from the args structure
        plan.ptra = from_args(offsetof(matmul_args_t, a));
        plan.ptrb = from_args(offsetof(matmul_args_t, b));
        plan.ptrc = from_args(offsetof(matmul_args_t, c));

        // Strides: a literal is baked in, a runtime one is loaded from args.
        // A strided load has no immediate-stride form, so transposed A's lda
        // is always a register (loaded below when literal)
        const bool ld_runtime = c_.ld_src == source_kind_t::runtime;
        bool starved = false;
        const auto take_int = [&] {
            const name_t n = pool.integers().pop();
            starved = starved || n == no_name;
            return Reg(n);
        };
        const Reg lda_reg = c_.trans_a && !ld_runtime ? take_int() : Reg {};
        plan.lda = ld_runtime ? from_args(offsetof(matmul_args_t, lda))
                : c_.trans_a  ? parameter_t(source_t(lda_reg))
                              : parameter_t(static_cast<int>(lda_bytes));
        plan.ldb = ld_runtime ? from_args(offsetof(matmul_args_t, ldb))
                              : parameter_t(static_cast<int>(ldb_bytes));
        plan.ldc = ld_runtime ? from_args(offsetof(matmul_args_t, ldc))
                              : parameter_t(static_cast<int>(ldc_bytes));

        // Sizes: a literal is baked in; every other kind is read by the backend
        // from args, or lives in a register loaded from args after planning.
        // `blocked` is a runtime size whose block the plan pins as given
        struct size_reg_t {
            Reg reg;
            size_t off;
        };
        std::vector<size_reg_t> size_regs;
        const auto make_size = [&](source_kind_t kind, size_t off,
                                       int64_t literal, int max) {
            if (kind == source_kind_t::literal)
                return size_parameter_t(static_cast<int>(literal));
            source_t src = from_args(off).source();
            if (c_.size_at == size_at_t::reg) {
                const Reg r = take_int();
                size_regs.push_back({r, off});
                src = source_t(r);
            }
            size_parameter_t size(src);
            if (kind == source_kind_t::less_eq) size.as_less_eq(max);
            return size;
        };
        plan.n = make_size(
                c_.shape_mn, offsetof(matmul_args_t, n), c_.n, c_.n_max);
        plan.k = make_size(
                c_.shape_k, offsetof(matmul_args_t, k), c_.k, c_.k_max);
        plan.m = make_size(
                c_.shape_mn, offsetof(matmul_args_t, m), c_.m, c_.m_max);

        // A blocked dimension pins its block: M in vectors, N and K as is
        if (c_.shape_mn == source_kind_t::blocked) {
            plan.blocks.mb = std::max(c_.m_blk / std::max(c_.vl, 1), 1);
            plan.blocks.nb = c_.n_blk;
        }
        if (c_.shape_k == source_kind_t::blocked) plan.blocks.kb = c_.k_blk;

        ok_ = !starved && backend.init(plan);
        if (!ok_) {
            ret();
            return;
        }

        persist.preserve();

        if (c_.trans_a && !ld_runtime) li(lda_reg, lda_bytes);
        for (const auto &sr : size_regs)
            ld(sr.reg, args, static_cast<int>(sr.off));

        backend();

        persist.restore();
        ret();
    }
};

} // namespace rvjit_bench

#endif
