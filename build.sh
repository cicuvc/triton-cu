#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export LLVM_SYSPATH=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/llvm-data/install
export TRITON_CACHE_PATH=${TRITON_CACHE_PATH:-$HOME/.triton/cache}
export TRITON_OFFLINE_BUILD=1
mkdir -p /tmp/triton_wheel

echo "=== Patching CMakeLists.txt ==="
python3 << 'PYEOF'
with open('CMakeLists.txt') as f: c = f.read()

# 1) Clang find_package after MLIR
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

# 2) Add CLANG_INCLUDE_DIRS
c = c.replace(
    'include_directories(${LLVM_INCLUDE_DIRS})',
    '''include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})''')

# 3) Remove AMD LLVM/MLIR libs
for line in ['    MLIRROCDLToLLVMIRTranslation\\n', '    MLIRGPUToROCDLTransforms\\n',
             '    LLVMAMDGPUCodeGen\\n', '    LLVMAMDGPUAsmParser\\n']:
    c = c.replace(line, '')

# 4) Add clang libs + LLVMMIRParser after LLVMNVPTXCodeGen
c = c.replace(
    '    LLVMNVPTXCodeGen\\n',
    '''    LLVMNVPTXCodeGen
    LLVMMIRParser

    clangCodeGen
    clangFrontend
    clangDriver
    clangBasic
    clangSerialization
    clangLex
    clangParse
    clangSema
    clangAST
''')

# 5) Add clang_compiler.cc after llvm.cc
c = c.replace(
    '                  ${PYTHON_SRC_PATH}/llvm.cc\\n                  ${PYTHON_SRC_PATH}/clang_compiler.cc\\n                  ${PYTHON_SRC_PATH}/specialize.cc)',
    '                  ${PYTHON_SRC_PATH}/llvm.cc\\n                  ${PYTHON_SRC_PATH}/clang_compiler.cc\\n                  ${PYTHON_SRC_PATH}/specialize.cc)')

# 6) Clang libs are built without RTTI; match for clang_compiler.cc
c += '''
set_source_files_properties(${PYTHON_SRC_PATH}/clang_compiler.cc
    PROPERTIES COMPILE_FLAGS "-fno-rtti")
'''

with open('CMakeLists.txt', 'w') as f: f.write(c)
print('  CMakeLists.txt patched')
PYEOF

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
echo "  PYTHONPATH=\"./python:./third_party/nvidia:./third_party/amd:\$PYTHONPATH\" python3 ..."
echo ""
echo "Undo patches: git checkout CMakeLists.txt"
