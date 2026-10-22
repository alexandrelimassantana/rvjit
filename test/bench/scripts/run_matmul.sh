#!/usr/bin/env bash
# Correctness-probes bench_matmul: with no args, runs its CI shape set
# (test/bench/matmul/inputs/test_matmul_ci); any args are forwarded as-is for
# ad-hoc single-shape/custom-batch exploration.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build=${RVJIT_BUILD_DIR:-$here/../../../build}

if [ "$#" -eq 0 ]; then
    set -- --mode=C --batch="$here/../matmul/inputs/test_matmul_ci"
fi

exec "$here/qemu-run.sh" "$build/test/bench/matmul/bench_matmul" "$@"
