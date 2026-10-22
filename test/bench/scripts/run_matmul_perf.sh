#!/usr/bin/env bash
# Soft-benchmarks bench_matmul over its full shape/tuning-knob sweep
# (test/bench/matmul/inputs/test_matmul_all). Manual/local use only -- never
# wired into CI, since QEMU timings aren't representative of real hardware
# (no self-hosted RV64 CI runner exists yet). Any args override the default
# --mode=P --batch=... invocation.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build=${RVJIT_BUILD_DIR:-$here/../../../build}

echo "note: QEMU timings are not representative of real hardware -- use" \
     "only to compare configs against each other, on this same machine," \
     "in the same run." >&2

if [ "$#" -eq 0 ]; then
    set -- --mode=P --batch="$here/../matmul/inputs/test_matmul_all"
fi

exec "$here/qemu-run.sh" "$build/test/bench/matmul/bench_matmul" "$@"
