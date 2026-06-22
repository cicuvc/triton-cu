#!/bin/bash
set -euo pipefail

# ============================================================
# Triton in-process CUDA compilation — build script
# ============================================================
# Requires:
#   - Self-compiled LLVM at LLVM_SYSPATH
#   - clang/clang++ as compiler
#   - Does NOT overwrite venv's installed triton
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LLVM_SYSPATH=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install
export TRITON_CACHE_PATH=${TRITON_CACHE_PATH:-$HOME/.triton/cache}
export TRITON_OFFLINE_BUILD=1
mkdir -p /tmp/triton_wheel

# Patch CMakeLists.txt: Clang + clang_compiler.cc
python3 << 'PYEOF'
with open('CMakeLists.txt') as f:
    c = f.read()

# Clang find_package after MLIR
c = c.replace(
    'find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})',
    '''find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

# Clang for in-process CUDA compilation
if(NOT CLANG_DIR)
  if(DEFINED LLVM_LIBRARY_DIR AND NOT "${LLVM_LIBRARY_DIR}" STREQUAL "")
    set(CLANG_DIR "${LLVM_LIBRARY_DIR}/cmake/clang")
  elseif(NOT "${LLVM_SYSPATH}" STREQUAL "")
    set(CLANG_DIR "${LLVM_SYSPATH}/lib/cmake/clang")
  endif()
endif()
find_package(Clang REQUIRED CONFIG PATHS ${CLANG_DIR})''')

# CLANG_INCLUDE_DIRS
c = c.replace(
    'include_directories(${LLVM_INCLUDE_DIRS})',
    '''include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})''')

# Remove AMD LLVM libs (our LLVM lacks AMDGPU target)
c = c.replace('    LLVMAMDGPUCodeGen\n', '')
c = c.replace('    LLVMAMDGPUAsmParser\n', '')

# Add clang libs + LLVMMIRParser (needed without AMDGPUAsmParser)
c = c.replace(
    '    LLVMNVPTXCodeGen\n',
    '''    LLVMNVPTXCodeGen
    LLVMMIRParser

    clangCodeGen clangFrontend clangDriver clangBasic
    clangSerialization clangLex clangParse clangSema clangAST''')

# Add clang_compiler.cc to sources
c = c.replace(
    '                  ${PYTHON_SRC_PATH}/llvm.cc\n                  ${PYTHON_SRC_PATH}/specialize.cc)',
    '                  ${PYTHON_SRC_PATH}/llvm.cc\n                  ${PYTHON_SRC_PATH}/clang_compiler.cc\n                  ${PYTHON_SRC_PATH}/specialize.cc)')

with open('CMakeLists.txt', 'w') as f:
    f.write(c)
print('CMakeLists.txt patched')
PYEOF

# Build
rm -rf build
CC=clang CXX=clang++ cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_BUILD_PROTON=OFF \
    -DTRITON_BUILD_UT=OFF \
    -DTRITON_CODEGEN_BACKENDS="nvidia" \
    -DLLVM_SYSPATH="${LLVM_SYSPATH}" \
    -DTRITON_CACHE_PATH="${TRITON_CACHE_PATH}" \
    -DTRITON_WHEEL_DIR=/tmp/triton_wheel \
    -B build \
    "$SCRIPT_DIR"

ln -sf build/compile_commands.json compile_commands.json

CC=clang CXX=clang++ ninja -C build triton

echo "=== BUILD COMPLETE ==="

cp build/libtriton.so python/triton/_C/libtriton.so
git checkout CMakeLists.txt

echo "cp build/libtriton.so python/triton/_C/libtriton.so"
echo "Undo: git checkout CMakeLists.txt"
