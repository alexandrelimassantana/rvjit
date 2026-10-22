#!/usr/bin/env bash
# Builds rvjit and runs its tests with the CI scripts (.github/automation):
# directly on an RV64 host, else inside the `rvjit-work` podman container,
# created from localhost/rvjit:latest, or from the CI image when that is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER="${RVJIT_CONTAINER:-rvjit-work}"
BUILD_DIR="${RVJIT_BUILD_DIR:-build}"
CI_DIR=.github/automation
CI_IMAGE=ubuntu:26.04

# Git runs on the host: the container mounts the live working tree
if [ ! -e "$LIB_DIR/third_party/xbyak_riscv/CMakeLists.txt" ]; then
    git -C "$LIB_DIR" submodule update --init
fi

if [ "$(uname -m)" = riscv64 ]; then
    cd "$LIB_DIR"
    "$CI_DIR/build.sh" "$BUILD_DIR"
    exec "$CI_DIR/test.sh" "$BUILD_DIR"
fi

if ! podman container exists "$CONTAINER"; then
    if podman image exists localhost/rvjit:latest; then
        image=localhost/rvjit:latest
    else
        image=$CI_IMAGE
    fi
    echo "Container '$CONTAINER' not found; creating one from $image" >&2
    podman run -d --network host --name "$CONTAINER" -v "$LIB_DIR":/workspace \
        "$image" sleep infinity
    if [ "$image" = "$CI_IMAGE" ]; then
        podman exec -w /workspace "$CONTAINER" "$CI_DIR/install.sh"
    fi
fi

if [ "$(podman inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
    podman start "$CONTAINER" >/dev/null
fi

podman exec -w /workspace "$CONTAINER" bash -c \
    '"$0/build.sh" "$1" && "$0/test.sh" "$1"' "$CI_DIR" "$BUILD_DIR"
