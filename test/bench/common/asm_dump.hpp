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

#ifndef RVJIT_TEST_BENCH_COMMON_ASM_DUMP_HPP
#define RVJIT_TEST_BENCH_COMMON_ASM_DUMP_HPP

#include <cstdint>
#include <cstdio>
#include <string>

namespace rvjit_bench {

/// Splits `code` into RVC-16-bit/RV-32-bit instruction words using the same
/// low-2-bits heuristic as `Xbyak_riscv::CodeGenerator::dump`, and writes
/// each as a `.insn` directive to `<dir>/<name>.s`
///
/// @note The written file assembles to a real ELF object whose text section
///     is marked as code (unlike a raw `.byte`/`.incbin` dump), so
///     `objdump -d` disassembles it including RVV mnemonics
inline void dump_kernel_asm(const std::string &dir, const std::string &name,
        const uint8_t *code, size_t size) {
    static int count = 0;
    const std::string path
            = dir + "/" + name + "_" + std::to_string(count++) + ".s";
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fputs(".text\n", f);
    size_t pos = 0;
    while (pos < size) {
        uint32_t v = code[pos];
        size_t n = (v & 3) == 3 ? 4 : 2;
        if (pos + n > size) break;
        uint32_t word = 0;
        for (size_t i = 0; i < n; i++)
            word |= static_cast<uint32_t>(code[pos + i]) << (8 * i);
        std::fprintf(
                f, ".insn %zu, 0x%0*x\n", n, static_cast<int>(n * 2), word);
        pos += n;
    }
    std::fclose(f);
    std::fprintf(stderr,
            "rvjit: wrote %s -- disassemble with:\n"
            "  riscv64-linux-gnu-as "
            "-march=rv64gcv_zfh_zvfh_zvfbfmin_zvfbfwma -o %s.o %s && "
            "riscv64-linux-gnu-objdump -d %s.o\n",
            path.c_str(), path.c_str(), path.c_str(), path.c_str());
}

} // namespace rvjit_bench

#endif
