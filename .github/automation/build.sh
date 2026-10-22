#!/usr/bin/env bash

#===============================================================================
# Copyright 2026 Barcelona Supercomputing Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Configures and builds rvjit: natively on an RV64 host, else cross-compiled
# with qemu-user as the test emulator (QEMU_LD_PREFIX overrides its sysroot).
#
# usage: build.sh [build-dir] [cmake-args...]

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build=${1:-build}
shift $(($# > 0))

args=(-DCMAKE_BUILD_TYPE=RelWithDebInfo)
if [ "$(uname -m)" != riscv64 ]; then
    sysroot=${QEMU_LD_PREFIX:-/usr/riscv64-linux-gnu}
    args+=(-DCMAKE_TOOLCHAIN_FILE="$root/cmake/toolchains/riscv64.cmake"
        "-DCMAKE_CROSSCOMPILING_EMULATOR=qemu-riscv64;-L;$sysroot")
fi

cmake -S "$root" -B "$build" "${args[@]}" "$@"
cmake --build "$build" -j"$(nproc)"
