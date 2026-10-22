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

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"

#include "rv64_jit_test_generator.hpp"
#include "rvjit/rvjit.hpp"

namespace rvjit {
namespace rvv {

using namespace Xbyak_riscv;

namespace {

struct gemm_args_t {
    const float *a, *b;
    float *c;
};

int read_vlen() {
    long vlenb = 0;
#if defined(__riscv)
    asm volatile("csrr %0, 0xc22" : "=r"(vlenb));
#endif
    return static_cast<int>(vlenb * 8);
}

} // namespace

/// The README's kernel computes a column-major f32 C = A * B
TEST(rvv_base_programming_model, readme_gemm) {
    static const int m = 21, n = 6, k = 9;
    vpu_t model;
    model.from_params(read_vlen(), 32 * 1024, 64, 4);

    bool planned = false;
    rv64_jit_probe_t kernel(1 << 16, [&](rv64_jit_test_generator_t &gen) {
        using namespace rvjit::linalg::matmul;

        auto regs = buffer::architecture_t::default_list();
        regs.integer.without({a0});
        base_programming_model_t pm(gen, model, regs);
        rvv::linalg::matmul::impl_t impl(pm);
        matmul_t gemm(impl);

        const auto arg = [](size_t off) {
            return parameter_t(source_t(pointer_t(a0, int(off))));
        };
        plan_t plan;
        plan.dta = plan.dtb = plan.dtc = type_t::f32;
        plan.m = size_parameter_t(m);
        plan.n = size_parameter_t(n);
        plan.k = size_parameter_t(k);
        plan.ptra = arg(offsetof(gemm_args_t, a));
        plan.ptrb = arg(offsetof(gemm_args_t, b));
        plan.ptrc = arg(offsetof(gemm_args_t, c));
        plan.lda = parameter_t(m * 4);
        plan.ldb = parameter_t(k * 4);
        plan.ldc = parameter_t(m * 4);

        planned = gemm.init(plan);
        if (planned) {
            pm.stack.preserve();
            gemm();
            pm.stack.restore();
        }
        gen.ret();
    });
    ASSERT_TRUE(kernel.create_kernel());
    ASSERT_TRUE(planned);

    std::vector<float> a(m * k), b(k * n), c(m * n, 0.f);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 7) - 3.f;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>(i % 5) - 2.f;
    gemm_args_t args {a.data(), b.data(), c.data()};
    run_timed(kernel, &args);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float want = 0.f;
            for (int p = 0; p < k; ++p)
                want += a[i + p * m] * b[p + j * k];
            ASSERT_EQ(c[i + j * m], want) << "at i=" << i << " j=" << j;
        }
}

} // namespace rvv
} // namespace rvjit
