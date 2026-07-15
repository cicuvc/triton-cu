// RUN: triton-opt %s -split-input-file -verify-diagnostics
//
// SHMLIR-01 positive test: Verifies that the ttg.extern_call ODS type constraint
// AnyTypeOf<[TT_Tensor, TTG_MemDescType]> accepts mixed tensor+memdesc operands
// in a single call. First operand is a tensor (registers), second is a memdesc
// (shared memory). Both pass the ODS constraint check per D-09.
//
// No expected-error markers = successful parse = PASS.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 2]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mixed_tensor_memdesc_extern_call(%a: tensor<32x64xf32, #blocked>, %b: !ttg.memdesc<4x4xf32, #shared, #smem>) {
    // CHECK: ttg.extern_call
    %result = ttg.extern_call %a, %b : (tensor<32x64xf32, #blocked>, !ttg.memdesc<4x4xf32, #shared, #smem>) -> tensor<32x64xf32, #blocked> { symbol = "test_fn", libpath = "test.cu" }
    tt.return
  }
}
