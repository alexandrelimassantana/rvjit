#!/usr/bin/env bash
# Drives bench_iterator/bench_memory_operand under qemu-user.
#
# usage: run_stream.sh iterator|memory_operand [args...]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build=${RVJIT_BUILD_DIR:-$here/../../../build}

if [ "$#" -lt 1 ]; then
    echo "usage: $(basename "$0") iterator|memory_operand [args...]" >&2
    exit 1
fi

kind=$1
shift
case "$kind" in
    iterator) bin=bench_iterator ;;
    memory_operand) bin=bench_memory_operand ;;
    *)
        echo "error: unknown kind '$kind' (expected 'iterator' or" \
             "'memory_operand')" >&2
        exit 1
        ;;
esac

exec "$here/qemu-run.sh" "$build/test/bench/stream/$bin" "$@"
