#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"

cmake -B "$BUILD_DIR" -S "$(dirname "$0")"

cmake --build "$BUILD_DIR" -- -j"$(nproc)"

"$BUILD_DIR/knn" "$@"
