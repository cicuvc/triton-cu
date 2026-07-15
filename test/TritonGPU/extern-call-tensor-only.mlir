// RUN: triton-opt %s -split-input-file -verify-diagnostics
//
// SHMLIR-01 regression test: Verifies that the ttg.extern_call ODS type constraint
// AnyTypeOf<[TT_Tensor, TTG_MemDescType]> accepts pure tensor operands without
// verifier error. This proves the relaxation to mixed types does not break the
// existing tensor-only parse path.
//
// All operands and results are tensors — no memdesc types here.
// No expected-error markers = successful parse = PASS.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tensor_only_extern_call(%a: tensor<32x64xf32, #blocked>, %b: tensor<32x64xf32, #blocked>) {
    // CHECK: ttg.extern_call
    %result = ttg.extern_call %a, %b : (tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked> { symbol = "elementwise_add", libpath = "tt_plugin.cu" }
    tt.return
  }
}
