#include "mlir/Analysis/Liveness.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {
#define GEN_PASS_DEF_TRITONNVIDIAGPUALLOCATENAMEDBARRIERSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

namespace {

constexpr int kNumHwBarriers = 16;
// Barrier 0 is used by all default `ttg.barrier` lowerings (bar.sync 0) and
// can be inserted by Membar anywhere in the kernel — never hand it out.
constexpr int kDefaultBarrierId = 0;
// Mirrors BarrierIndex in ConvertWarpSpecializeToLLVM.cpp: the switch-loop
// barrier (1) and per-partition barriers (2 + partitionIdx) are reserved
// while a ttg.warp_specialize op is live.
constexpr int kSwitchLoopBarrierId = 1;
constexpr int kFirstPartitionBarrierId = 2;

// Identity of the warp group an op executes in: the direct child region of
// the enclosing ttg.warp_specialize (default region or one partition region),
// or nullptr for ops outside any warp_specialize (the top-level default warp
// group of the function).
Region *warpGroupRegionOf(Operation *op) {
  Region *region = op->getParentRegion();
  while (region) {
    Operation *parent = region->getParentOp();
    if (isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(parent))
      return region;
    region = parent->getParentRegion();
  }
  return nullptr;
}

struct NamedBarrierAllocator {
  explicit NamedBarrierAllocator(ModuleOp mod) : mod(mod) {}

  ModuleOp mod;
  DenseMap<Operation *, size_t> operationId;

  struct TokenInfo {
    ttg::AllocNamedBarrierOp alloc;
    // [min, max] operation ids where the token is live.
    size_t liveMin = 0, liveMax = 0;
    // Direct child regions of warp_specialize ops touched by alloc/uses,
    // keyed by the enclosing warp_specialize op.
    DenseMap<Operation *, DenseSet<Region *>> wsRegions;
    int assignedId = -1;
  };

  SmallVector<TokenInfo> tokens;

  // Reserved IDs per warp_specialize op, with the op subtree's id interval.
  struct WsReservation {
    size_t subtreeMin, subtreeMax;
    int numPartitions;
  };
  SmallVector<WsReservation> reservations;

  void assignOperationIds() {
    operationId.reserve(1024);
    mod.walk<WalkOrder::PostOrder>(
        [&](Operation *op) { operationId[op] = operationId.size(); });
  }

  void collectTokens() {
    Liveness liveness(mod);
    mod.walk([&](ttg::AllocNamedBarrierOp alloc) {
      Value token = alloc.getResult();
      TokenInfo info;
      info.alloc = alloc;

      if (token.use_empty()) {
        alloc.emitRemark("named barrier is never used; no ID assigned");
        return;
      }

      // Liveness interval from MLIR Liveness over post-order ids. Values
      // defined outside and used inside nested regions stay live across the
      // parent op because parents get larger ids than all their children.
      size_t minId = std::numeric_limits<size_t>::max();
      size_t maxId = 0;
      for (Operation *liveOp : liveness.resolveLiveness(token)) {
        minId = std::min(minId, operationId[liveOp]);
        maxId = std::max(maxId, operationId[liveOp]);
      }
      info.liveMin = minId;
      info.liveMax = maxId + 1;

      // Warp-group regions touched by the alloc itself and every use.
      auto noteRegion = [&](Operation *op) {
        if (Operation *ws = op->getParentOfType<ttg::WarpSpecializeOp>()) {
          info.wsRegions[ws].insert(warpGroupRegionOf(op));
        }
      };
      noteRegion(alloc);
      for (OpOperand &use : token.getUses())
        noteRegion(use.getOwner());

      tokens.push_back(std::move(info));
    });
  }

  void collectReservations() {
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      WsReservation res;
      size_t minId = operationId[ws];
      ws->walk<WalkOrder::PostOrder>(
          [&](Operation *op) { minId = std::min(minId, operationId[op]); });
      res.subtreeMin = minId;
      res.subtreeMax = operationId[ws] + 1;
      res.numPartitions = static_cast<int>(ws.getPartitionNumWarps().size());
      reservations.push_back(res);
    });
  }

  bool intervalsOverlap(size_t aMin, size_t aMax, size_t bMin, size_t bMax) {
    return aMin < bMax && bMin < aMax;
  }

  // Two tokens interfere (cannot share a hardware ID) if their live ranges
  // overlap, or if they touch different warp-group regions of the same
  // warp_specialize op (those regions execute concurrently even though their
  // post-order id intervals are disjoint).
  bool tokensInterfere(const TokenInfo &a, const TokenInfo &b) {
    if (intervalsOverlap(a.liveMin, a.liveMax, b.liveMin, b.liveMax))
      return true;
    for (auto &[ws, regionsA] : a.wsRegions) {
      auto it = b.wsRegions.find(ws);
      if (it == b.wsRegions.end())
        continue;
      const DenseSet<Region *> &regionsB = it->second;
      if (regionsA != regionsB)
        return true;
    }
    return false;
  }

  // Infer or validate the arrival count for one token.
  LogicalResult resolveCount(TokenInfo &info) {
    ttg::AllocNamedBarrierOp alloc = info.alloc;
    Value token = alloc.getResult();

    // Distinct warp groups among all uses (region identity, not warp count —
    // two partitions may coincidentally have the same warp count).
    DenseSet<Region *> groups;
    bool hasArrive = false;
    for (OpOperand &use : token.getUses()) {
      Operation *owner = use.getOwner();
      groups.insert(warpGroupRegionOf(owner));
      hasArrive |= isa<ttg::NamedBarrierArriveOp>(owner);
    }

    // Inferred union of participating threads across all use sites.
    int inferredUnion = 0;
    {
      DenseMap<Region *, int> perGroupWarps;
      for (OpOperand &use : token.getUses()) {
        Region *g = warpGroupRegionOf(use.getOwner());
        perGroupWarps[g] = ttg::lookupNumWarps(use.getOwner());
      }
      for (auto &[g, w] : perGroupWarps)
        inferredUnion += w * 32;
    }

    if (auto countAttr = alloc.getCountAttr()) {
      int explicitCount = countAttr.getInt();
      if (explicitCount != inferredUnion) {
        alloc.emitWarning()
            << "explicit named barrier count " << explicitCount
            << " differs from the union of threads at its use sites ("
            << inferredUnion << "); trusting the explicit value";
      }
      return success();
    }

    if (groups.size() > 1) {
      return alloc.emitError()
             << "named barrier without explicit count is used across "
                "multiple warp groups; cross-warp-group barriers require an "
                "explicit `count`";
    }
    if (hasArrive) {
      return alloc.emitError()
             << "named barrier without explicit count is used by "
                "ttg.named_barrier_arrive; split arrive/wait barriers "
                "require an explicit `count`";
    }
    alloc.setCountAttr(IntegerAttr::get(IntegerType::get(mod.getContext(), 32),
                                        inferredUnion));
    return success();
  }

  // First-fit graph coloring over hardware IDs 1..15.
  LogicalResult assignIds() {
    // Sort by live-range start for deterministic first-fit.
    llvm::sort(tokens, [](const TokenInfo &a, const TokenInfo &b) {
      return a.liveMin < b.liveMin;
    });

    for (size_t i = 0; i < tokens.size(); ++i) {
      TokenInfo &tok = tokens[i];
      bool forbidden[kNumHwBarriers] = {};
      forbidden[kDefaultBarrierId] = true;

      // Reserved IDs of every warp_specialize whose interval overlaps.
      for (const WsReservation &res : reservations) {
        if (!intervalsOverlap(tok.liveMin, tok.liveMax, res.subtreeMin,
                              res.subtreeMax))
          continue;
        forbidden[kSwitchLoopBarrierId] = true;
        for (int p = 0; p < res.numPartitions; ++p) {
          int id = kFirstPartitionBarrierId + p;
          if (id < kNumHwBarriers)
            forbidden[id] = true;
        }
      }

      // IDs taken by interfering, already-colored tokens.
      for (size_t j = 0; j < i; ++j) {
        if (tokens[j].assignedId >= 0 && tokensInterfere(tokens[j], tok))
          forbidden[tokens[j].assignedId] = true;
      }

      int id = -1;
      for (int cand = 1; cand < kNumHwBarriers; ++cand) {
        if (!forbidden[cand]) {
          id = cand;
          break;
        }
      }
      if (id < 0) {
        return tok.alloc.emitError()
               << "failed to allocate a hardware named barrier: all "
               << (kNumHwBarriers - 1)
               << " non-reserved barrier IDs are exhausted (live ranges "
                  "interfere); reduce the number of concurrently live named "
                  "barriers";
      }
      tok.assignedId = id;
      tok.alloc.setBarrierIdAttr(
          IntegerAttr::get(IntegerType::get(mod.getContext(), 32), id));
    }
    return success();
  }

  LogicalResult run() {
    assignOperationIds();
    collectTokens();
    if (tokens.empty())
      return success();
    collectReservations();
    for (TokenInfo &tok : tokens)
      if (failed(resolveCount(tok)))
        return failure();
    return assignIds();
  }
};

struct TritonNvidiaGPUAllocateNamedBarriersPass
    : public impl::TritonNvidiaGPUAllocateNamedBarriersPassBase<
          TritonNvidiaGPUAllocateNamedBarriersPass> {
  void runOnOperation() override {
    if (failed(NamedBarrierAllocator{getOperation()}.run()))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
