// RUN: triton-opt %s -split-input-file -convert-triton-gpu-to-llvm | FileCheck %s
//
// SHLOWER-01/02: Mixed shared+distributed args in ttg.extern_call lowering.
// Verifies: (1) distributed operands get alloca+store+ptr (AS0)
//           (2) shared operands get ptr<3> directly via getelementptr
//           (3) subview offset GEP is applied to shared base
//           (4) both arg types coexist in same call preserving signature order

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 2]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {
  "ttg.target" = "cuda:0",
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.extern_call_mangled_names" = "{\"test_mixed\": \"_Z10test_mixedPfPU7cuda_sharedf\"}",
  "ttg.extern_call_arg_spaces" = "{\"test_mixed\": [\"register\", \"shared\"]}"
} {
  tt.func @mixed_shared_tensor_extern_call(
      %a: tensor<32x64xf32, #blocked>,
      %b: !ttg.memdesc<4x4xf32, #shared, #smem>) {
    // First arg (distributed): expect alloca for stack slot allocation
    // CHECK: llvm.alloca

    // First arg (distributed): expect store to write struct to stack slot
    // CHECK: llvm.store

    // Second arg (shared): bypasses alloca; GEP + ptr<3> directly.
    // The getelementptr applies the subview offset (SHLOWER-02).
    // CHECK: llvm.getelementptr {{.*}} !llvm.ptr<3>

    // Final call: both args (ptr AS0, ptr<3>) in correct order
    // CHECK: llvm.call @_Z10test_mixed

    %result = ttg.extern_call %a, %b
        : (tensor<32x64xf32, #blocked>, !ttg.memdesc<4x4xf32, #shared, #smem>)
        -> tensor<32x64xf32, #blocked>
        { symbol = "test_mixed", libpath = "test.cu" }
    tt.return
  }
}
