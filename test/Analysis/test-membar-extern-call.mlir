// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s
//
// Verifies that ttg.extern_call declares MemRead+MemWrite effects on its
// shared-memory (memdesc) operands, so Membar inserts the barriers required
// for cross-thread visibility of the called CUDA device function's accesses.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [1, 0], [2, 2]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// CHECK-LABEL: extern_call_write_then_load
tt.func @extern_call_write_then_load() {
  %smem = ttg.local_alloc : () -> !ttg.memdesc<4x4xf32, #shared, #smem, mutable>
  // The extern call may write the buffer; the subsequent local_load must be
  // separated by a barrier.
  // CHECK: ttg.extern_call
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.local_load
  ttg.extern_call %smem
      : (!ttg.memdesc<4x4xf32, #shared, #smem, mutable>) -> ()
      { symbol = "mutate_shared", libpath = "test.cu" }
  %v = ttg.local_load %smem : !ttg.memdesc<4x4xf32, #shared, #smem, mutable> -> tensor<4x4xf32, #blocked>
  tt.return
}

// CHECK-LABEL: store_then_extern_call_read
tt.func @store_then_extern_call_read(%x: tensor<4x4xf32, #blocked>) {
  %smem = ttg.local_alloc : () -> !ttg.memdesc<4x4xf32, #shared, #smem, mutable>
  // The extern call may read the buffer; the preceding local_store must be
  // visible to it across threads.
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttg.extern_call
  ttg.local_store %x, %smem : tensor<4x4xf32, #blocked> -> !ttg.memdesc<4x4xf32, #shared, #smem, mutable>
  ttg.extern_call %smem
      : (!ttg.memdesc<4x4xf32, #shared, #smem, mutable>) -> ()
      { symbol = "read_shared", libpath = "test.cu" }
  tt.return
}

// CHECK-LABEL: extern_call_register_only
tt.func @extern_call_register_only(%a: tensor<4x4xf32, #blocked>, %b: tensor<4x4xf32, #blocked>) {
  // Register-tensor operands have no memory effects — no barrier.
  // CHECK: ttg.extern_call
  // CHECK-NEXT: tt.return
  %r = ttg.extern_call %a, %b
      : (tensor<4x4xf32, #blocked>, tensor<4x4xf32, #blocked>)
      -> tensor<4x4xf32, #blocked>
      { symbol = "elementwise_add", libpath = "test.cu" }
  tt.return
}

}
