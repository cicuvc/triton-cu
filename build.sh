#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export LLVM_SYSPATH=/mnt/data/llvm-triton/install/
export TRITON_CACHE_PATH=${TRITON_CACHE_PATH:-$HOME/.triton/cache}
export TRITON_OFFLINE_BUILD=1
mkdir -p /tmp/triton_wheel

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
    -DTRITON_WHEEL_DIR=/tmp/triton_wheel \
    -B build \
    .

ln -sf build/compile_commands.json compile_commands.json

CC=clang CXX=clang++ ninja -C build triton

echo ""
echo "=== BUILD COMPLETE ==="
echo "  cp build/libtriton.so python/triton/_C/libtriton.so"
echo "  PYTHONPATH=\"./python:./third_party/nvidia:\$PYTHONPATH\" python3 ..."