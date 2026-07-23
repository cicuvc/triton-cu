# Phase 4: C++ Templates + Clang AST Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-12
**Phase:** 4-C++ Templates + Clang AST Foundation
**Areas discussed:** SharedLinearLayout template shape, SharedTensor storage & accessors, Scope of evaluate()/swizzle math, Verification harness

---

## Area selection

| Option | Description | Selected |
|--------|-------------|----------|
| SharedLinearLayout template shape | Flat top-level vs factory-nested BasisGroup | ✓ |
| SharedTensor storage & accessors | C++ storage + read/write model | ✓ |
| Scope of evaluate()/swizzle math | Implement + parity test now vs defer to P7 | ✓ |
| Verification harness (no GPU yet) | How to prove 5 success criteria pre-integration | ✓ |
| TypeInspector variant integration | How SharedTensorParameter joins inference variant | (not selected — left to planner) |

---

## SharedLinearLayout template shape

| Option | Description | Selected |
|--------|-------------|----------|
| Flat top-level template | `SharedLinearLayout<OffsetBases{...}, BlockBases{...}, Align>`, dedicated carrier structs, matches criteria syntax | ✓ |
| Reuse BasisGroup, nested factory | `SharedTensorLayout<Shape>::SharedLinearLayout<...>` reusing existing BasisGroup | |
| Flat, but share IntTuple/BasisGroup primitives | Flat public shape, shared low-level primitives | |

**User's choice:** Flat top-level template.
**Notes:** Follow-up question on basis-row representation was aborted; the recommended default (rows = length-`rank` logical-dim vectors, for MLIR round-trip parity) was locked, since it is the only representation that round-trips against MLIR's multi-output shared layout.

---

## SharedTensor storage & accessors

| Option | Description | Selected |
|--------|-------------|----------|
| T* data member + operator[] returning T& | Pointer member, index→offset via layout | |
| Minimal shell, defer accessors to Phase 6 | Just type params, defer semantics | |
| Byte pointer + explicit load/store methods | uintptr_t + load/store | |
| (Custom) zero-length array + operator() | `T data[]` zero-length array; `operator()` → layout.evaluate() → offset → `T&` | ✓ |

**User's choice:** Custom — zero-length array `T data[]` inside SharedTensor; overload `operator()`, input runs through the layout's `evaluate()` to get an offset, returning `dtype&`.
**Notes:** Zero storage cost; base pointer at struct address maps cleanly to `ptr addrspace(3)` in Phase 6. Directly implies `evaluate()` must exist and be correct this phase.

---

## Scope of evaluate()/swizzle math

| Option | Description | Selected |
|--------|-------------|----------|
| Implement evaluate() + parity unit test now | Full impl + bit-identical MLIR parity check this phase | ✓ |
| Implement evaluate(), defer parity check to P7 | Smoke-test only now | |
| Implement evaluate() sharing MLIR's formula | Port MLIR offset formula into C++ | |

**User's choice:** Implement evaluate() + parity unit test now.
**Notes:** Front-loads the STATE.md swizzle-parity risk (C++ `evaluate()` must be bit-identical to MLIR `LinearLayout({offsetBases,blockBases})` at gluon_ir.cc:102-103) instead of deferring to Phase 7.

---

## Verification harness (no GPU yet)

| Option | Description | Selected |
|--------|-------------|----------|
| Python pytest via llvm bindings | Construct SharedTensorParameter, drive inference on synthetic .cu, assert round-trip/resolution/parity | ✓ |
| Standalone C++ / gtest binary | New C++ test target | |
| lit + FileCheck on -ast-dump | Golden AST-dump test | |
| pytest + standalone nvcc compile check | Combine binding pytest with nvcc compile of .cu | |

**User's choice:** Python pytest via llvm bindings.
**Notes:** GPU-free, runs under `make test-unit`; style mirrors `test_extern_call.py` without `@gluon.jit`. Open research thread: how to invoke C++ `evaluate()` from Python for the parity check (captured in CONTEXT.md deferred/research).

---

## Agent's Discretion

- TypeInspector variant integration (5th area, not selected for discussion): how `SharedTensorParameter` joins the existing `variant<nullptr_t, TensorParameter, TupleType>` inference flow in `DispatchTypeParsing` — left to planner/researcher (SHAST-03 requires same-flow resolution; exact plumbing open).

## Deferred Ideas

- Research: mechanism to invoke C++ `evaluate()` from Python for the D-07 parity test.
- `PaddedSharedLayout` support — out of scope for v1.1.
- Dynamic / `extern __shared__` variable-size shared allocation — out of scope.
- Returning `shared_memory_descriptor` from `gl.call()` (SHRET-01) — future milestone.
