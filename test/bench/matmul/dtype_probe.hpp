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

#ifndef RVJIT_TEST_BENCH_MATMUL_DTYPE_PROBE_HPP
#define RVJIT_TEST_BENCH_MATMUL_DTYPE_PROBE_HPP

#include <csetjmp>
#include <csignal>

#include "bench/common/dtype.hpp"
#include "matmul_case.hpp"
#include "rvjit/rvjit.hpp"

namespace rvjit_bench {

namespace detail {

inline sigjmp_buf &probe_jmpbuf() {
    static sigjmp_buf buf;
    return buf;
}

inline void on_sigill(int) {
    siglongjmp(probe_jmpbuf(), 1);
}

/// A one-shot kernel executing the float instructions a dtype needs
struct probe_kernel_t : public Xbyak_riscv::CodeGenerator {
    explicit probe_kernel_t(bool bf16) : Xbyak_riscv::CodeGenerator(4096) {
        using namespace Xbyak_riscv;
        vsetvli(x0, x0, SEW::e16, LMUL::m1);
        flh(ft0, sp, 0);
        if (bf16)
            vfwmaccbf16_vv(v2, v0, v1);
        else
            vfmacc_vv(v2, v0, v1);
        ret();
        ready();
    }

    /// Whether the emulator or hardware executes the kernel
    bool runs() const {
        struct sigaction sa {};
        struct sigaction old {};
        sa.sa_handler = on_sigill;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGILL, &sa, &old);
        bool ok = true;
        if (sigsetjmp(probe_jmpbuf(), 1) == 0)
            reinterpret_cast<void (*)()>(const_cast<uint8_t *>(getCode()))();
        else
            ok = false;
        sigaction(SIGILL, &old, nullptr);
        return ok;
    }
};

} // namespace detail

/// Whether the machine executes the instructions the case's dtypes need
///
/// @note Half precision needs Zvfh and bf16 Zvfbfwma, which older emulators
///     lack
inline bool dtypes_supported(const matmul_case_t &c) {
    const auto uses = [&](dtype_t dt) {
        return c.dta == dt || c.dtb == dt || c.dtc == dt || c.dtacc == dt;
    };
    static const bool f16_ok = detail::probe_kernel_t(false).runs();
    static const bool bf16_ok = detail::probe_kernel_t(true).runs();
    return (!uses(dtype_t::f16) || f16_ok) && (!uses(dtype_t::bf16) || bf16_ok);
}

} // namespace rvjit_bench

#endif
