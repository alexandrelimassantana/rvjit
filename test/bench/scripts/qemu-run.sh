#!/usr/bin/env bash
# Runs an RV64 bench binary: directly on an RV64 host, else under qemu-user with
# the sysroot CI's CMAKE_CROSSCOMPILING_EMULATOR uses (QEMU_LD_PREFIX overrides
# it, see .github/automation/build.sh).
set -euo pipefail
if [ "$(uname -m)" = riscv64 ]; then
    if [ -n "${RVJIT_QEMU_VLEN:-}" ]; then
        echo "note: RVJIT_QEMU_VLEN ignored on an RV64 host" >&2
    fi
    exec "$@"
fi
# QEMU_CPU picks the emulated CPU; without it, RVJIT_QEMU_VLEN=128|256|512|1024
# picks the vector length of qemu's `max` CPU. An Ubuntu >= 25.10 sysroot is
# built for RVA23, so a QEMU_CPU without Zcb and the other RVA23 extensions SIGILLs
cpu=()
if [ -n "${QEMU_CPU:-}" ]; then
    if [ -n "${RVJIT_QEMU_VLEN:-}" ]; then
        echo "note: RVJIT_QEMU_VLEN ignored, QEMU_CPU is set" >&2
    fi
elif [ -n "${RVJIT_QEMU_VLEN:-}" ]; then
    cpu=(-cpu "max,vlen=${RVJIT_QEMU_VLEN},vext_spec=v1.0")
fi
exec qemu-riscv64 -L "${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}" \
    ${cpu[@]+"${cpu[@]}"} "$@"
