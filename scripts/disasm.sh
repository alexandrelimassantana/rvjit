#!/usr/bin/env bash
# Assembles a rvjit bench --dump-asm .s file (see test/bench/common/asm_dump.hpp)
# into a real RV64 object, objdumps it to stdout, then deletes the
# intermediary .o -- wraps the manual `riscv64-linux-gnu-as`/`objdump` pair
# dump_kernel_asm() itself prints into one step.
#
# AS/OBJDUMP override the tools (native `as`/`objdump` on an RV64 host), MARCH
# the ISA string (the default names Zvfh/Zvfbfmin/Zvfbfwma so that f16 and
# bf16 conversions and FMAs decode instead of printing as raw .insn words).
#
# usage: disasm.sh <path/to/dump.s>
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $(basename "$0") <path/to/dump.s>" >&2
    exit 1
fi

prefix=riscv64-linux-gnu-
if [ "$(uname -m)" = riscv64 ]; then
    prefix=
fi
as=${AS:-${prefix}as}
objdump=${OBJDUMP:-${prefix}objdump}
march=${MARCH:-rv64gcv_zfh_zvfh_zvfbfmin_zvfbfwma}

src="$1"
obj="${src}.o"

trap 'rm -f "$obj"' EXIT

"$as" -march="$march" -o "$obj" "$src"
"$objdump" -d "$obj"
