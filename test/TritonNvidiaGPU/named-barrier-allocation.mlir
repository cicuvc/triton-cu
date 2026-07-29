// RUN: triton-opt %s --triton-nvidia-gpu-allocate-named-barriers | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // Single token, sync-only in the default warp group: count is inferred as
  // num-warps * 32 and the first non-reserved ID (1) is assigned.
  // CHECK-LABEL: @basic
  tt.func @basic() {
    // CHECK: %[[B:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    // CHECK-SAME: count = 128 : i32
    %bar = ttg.alloc_named_barrier
    // CHECK: ttg.named_barrier_sync %[[B]]
    ttg.named_barrier_sync %bar
    tt.return
  }

  // Explicit count is preserved and enables arrive usage.
  // CHECK-LABEL: @explicit_count
  tt.func @explicit_count() {
    // CHECK: %[[B:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    // CHECK-SAME: count = 64 : i32
    %bar = ttg.alloc_named_barrier {count = 64 : i32}
    // CHECK: ttg.named_barrier_arrive %[[B]]
    ttg.named_barrier_arrive %bar
    // CHECK: ttg.named_barrier_sync %[[B]]
    ttg.named_barrier_sync %bar
    tt.return
  }

  // Disjoint live ranges share a hardware ID (like register allocation).
  // CHECK-LABEL: @disjoint_liveness_shares_id
  tt.func @disjoint_liveness_shares_id() {
    // CHECK: %[[A:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    %a = ttg.alloc_named_barrier
    ttg.named_barrier_sync %a
    // CHECK: %[[B:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    %b = ttg.alloc_named_barrier
    ttg.named_barrier_sync %b
    tt.return
  }

  // Overlapping live ranges get distinct IDs.
  // CHECK-LABEL: @overlapping_liveness_distinct_ids
  tt.func @overlapping_liveness_distinct_ids() {
    // CHECK: %[[A:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    %a = ttg.alloc_named_barrier
    // CHECK: %[[B:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 2 : i32
    %b = ttg.alloc_named_barrier
    ttg.named_barrier_sync %a
    ttg.named_barrier_sync %b
    tt.return
  }

  // Warp specialization reserves IDs 1 (switch loop) and 2..2+P-1
  // (partitions). Tokens inside the ws region get IDs past the reserved
  // range; partition tokens infer the partition's thread count; tokens in
  // concurrently executing regions interfere.
  // CHECK-LABEL: @warp_specialize_reserved_ids
  tt.func @warp_specialize_reserved_ids() {
    ttg.warp_specialize()
    default {
      // Default region: count = ttg.num-warps * 32 = 128; reserved IDs are
      // {1, 2}, so the first free ID is 3.
      // CHECK: %[[D:.*]] = ttg.alloc_named_barrier
      // CHECK-SAME: barrier_id = 3 : i32
      // CHECK-SAME: count = 128 : i32
      %d = ttg.alloc_named_barrier
      ttg.named_barrier_sync %d
      ttg.warp_yield
    }
    partition0() num_warps(2) {
      // Partition: count = 2 * 32 = 64; runs concurrently with the default
      // region, so it cannot share the default token's ID.
      // CHECK: %[[P:.*]] = ttg.alloc_named_barrier
      // CHECK-SAME: barrier_id = 4 : i32
      // CHECK-SAME: count = 64 : i32
      %p = ttg.alloc_named_barrier
      ttg.named_barrier_sync %p
      ttg.warp_return
    } : () -> ()
    tt.return
  }

  // A token that dies before the warp_specialize op may reuse reserved IDs.
  // CHECK-LABEL: @token_outside_ws_reuses_reserved
  tt.func @token_outside_ws_reuses_reserved() {
    // CHECK: %[[E:.*]] = ttg.alloc_named_barrier
    // CHECK-SAME: barrier_id = 1 : i32
    %e = ttg.alloc_named_barrier
    ttg.named_barrier_sync %e
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(2) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
