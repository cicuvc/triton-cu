#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${LLVM_SYSPATH:-}" ]; then
  echo "ERROR: LLVM_SYSPATH is not set." >&2
  echo "  export LLVM_SYSPATH=/path/to/llvm/install" >&2
  echo "  (e.g. /mnt/data/llvm-triton/install or /usr/local/llvm)" >&2
  exit 1
fi
export LLVM_SYSPATH
export TRITON_CACHE_PATH=${TRITON_CACHE_PATH:-$HOME/.triton/cache}
export TRITON_OFFLINE_BUILD=1

WHEEL_DIR=${TRITON_WHEEL_DIR:-/tmp/triton_wheel}
mkdir -p "$WHEEL_DIR"

echo "=== Building ==="
rm -rf build
CC=clang CXX=clang++ cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_LINKER=ld.lld \
    -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_BUILD_PROTON=OFF \
    -DTRITON_BUILD_UT=OFF \
    -DTRITON_CODEGEN_BACKENDS="nvidia" \
    -DLLVM_SYSPATH="${LLVM_SYSPATH}" \
    -DTRITON_CACHE_PATH="${TRITON_CACHE_PATH}" \
    -DTRITON_WHEEL_DIR="$WHEEL_DIR" \
    -B build \
    .

ln -sf build/compile_commands.json compile_commands.json

CC=clang CXX=clang++ ninja -C build triton

echo ""
echo "=== BUILD COMPLETE ==="
ln -sf "$(pwd)/build/libtriton.so" python/triton/_C/libtriton.so
echo "  PYTHONPATH=\"./python:./third_party/nvidia:\$PYTHONPATH\" python3 ..."