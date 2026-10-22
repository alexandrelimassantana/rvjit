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

#ifndef RVJIT_TEST_BENCH_COMMON_CSR_HPP
#define RVJIT_TEST_BENCH_COMMON_CSR_HPP

#include <cstdint>

namespace rvjit_bench {

/// Reads the `vlenb` CSR (0xc22): vector register length in bytes
///
/// @note Numeric CSR address, not the symbolic "vlenb" name: the assembler
///     only recognizes that name when -march enables V, but this code
///     doesn't need to
/// @note Zero on a build that never executes on RV64 (e.g. this file
///     compiled for a cross-compilation host); real hardware/qemu-user
///     execution is required for a real value
inline int64_t read_vlenb() {
    long v = 0;
#if defined(__riscv)
    asm volatile("csrr %0, 0xc22" : "=r"(v));
#endif
    return v;
}

/// Reads the `instret` CSR (0xc02): retired instruction count
///
/// @note Requires the Zicntr extension's counters to be readable from U-mode,
///     not verified against every qemu version or CPU model
inline int64_t read_instret() {
    long v = 0;
#if defined(__riscv)
    asm volatile("csrr %0, 0xc02" : "=r"(v));
#endif
    return v;
}

} // namespace rvjit_bench

#endif
