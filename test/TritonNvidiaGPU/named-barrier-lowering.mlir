// RUN: triton-opt %s --triton-nvidia-gpu-allocate-named-barriers --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @lower_sync
  tt.func @lower_sync() {
    // The alloc op produces no runtime code; sync lowers to nvvm.barrier
    // with the assigned id and inferred count.
    // CHECK-NOT: ttg.alloc_named_barrier
    // CHECK: nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}}
    %bar = ttg.alloc_named_barrier
    ttg.named_barrier_sync %bar
    tt.return
  }

  // CHECK-LABEL: @lower_arrive
  tt.func @lower_arrive() {
    // CHECK: nvvm.barrier.arrive id = %{{.*}} number_of_threads = %{{.*}}
    %bar = ttg.alloc_named_barrier {count = 64 : i32}
    ttg.named_barrier_arrive %bar
    tt.return
  }
}
