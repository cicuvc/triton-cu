// RUN: triton-opt %s --triton-nvidia-gpu-allocate-named-barriers --verify-diagnostics --split-input-file

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @arrive_requires_explicit_count() {
    // expected-error@+1 {{named barrier without explicit count is used by ttg.named_barrier_arrive; split arrive/wait barriers require an explicit `count`}}
    %bar = ttg.alloc_named_barrier
    ttg.named_barrier_arrive %bar
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @cross_region_requires_explicit_count() {
    // expected-error@+1 {{named barrier without explicit count is used across multiple warp groups; cross-warp-group barriers require an explicit `count`}}
    %bar = ttg.alloc_named_barrier
    ttg.named_barrier_sync %bar
    ttg.warp_specialize(%bar)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.named_barrier) num_warps(2) {
      ttg.named_barrier_sync %arg0
      ttg.warp_return
    } : (!ttg.named_barrier) -> ()
    tt.return
  }
}
