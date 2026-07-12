# Phase 1: Seam & Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-11
**Phase:** 1-seam-cleanup
**Areas discussed:** Inference hook contract, Single-parse strategy (INFER-07), f64/fp64 handling (BUG-02), Graceful degradation (INFER-06)

---

## Inference Hook Contract

### Form of the hook
| Option | Description | Selected |
|--------|-------------|----------|
| Single callable in codegen_fns dict | A Python callable under a dict key, mirroring convert_custom_types/min_dot_size | |
| Stateful object with methods | Small object/namespace with methods so semantic-time inference and llir compilation share one handle | ✓ |
| You decide / describe | — | |

**User's choice:** Stateful object with methods.

### Method surface
| Option | Description | Selected |
|--------|-------------|----------|
| infer_result + compile_bitcode | Two methods sharing the same parsed .cu | ✓ |
| Single combined call, stash bitcode | One call does inference and stashes bitcode | |
| You decide / describe | — | |

**User's choice:** infer_result(func, arg_params, use_fast_math) + compile_bitcode(requests).

### LLVMContext layering challenge
| Option | Description | Selected |
|--------|-------------|----------|
| Cache parse, emit bitcode later | Cache resolved FunctionDecls/return params without bitcode; emit later | |
| Cache whole CUDACompiler instance | Cache the live coroutine + its context | ✓ |
| Document two-parse, defer caching | Inference-only now, document + measure two-parse cost | |
| You decide / describe | — | |

**User's choice:** Cache the whole CUDACompiler.
**Notes:** User rationale — the coroutine is *more* stable than caching decls. Once the ASTConsumer flow completes, clang's AST is no longer used, internal pointers are invalidated, and the ASTContext is torn down; cached FunctionDecls would then be invalid and unable to support further type deduction/inspection. Parking the flow inside HandleTranslationUnit (suspended) until LLVM IR generation truly begins is the safest.

### Coroutine lifetime
| Option | Description | Selected |
|--------|-------------|----------|
| Per-compile: suspend then consume at llir | Created+suspended at semantic time, resumed+consumed at llir, then torn down | ✓ |
| Process-global slot, rebuild after consume | Global slot keyed by source, rebuilt after each consume | |
| You decide / describe | — | |

**User's choice:** Per-compile: suspend then consume at llir.
**Notes:** Refines the earlier "process-global cached by source" framing — the *live* coroutine is per-compile; disk cache handles repeated compiles.

---

## Single-parse Strategy (INFER-07)

### Handoff mechanism
| Option | Description | Selected |
|--------|-------------|----------|
| Stash in metadata dict | Store the suspended CUDACompiler in the compile metadata dict | ✓ |
| Global registry keyed by source | Process-global registry both stages look up | |
| You decide / describe | — | |

**User's choice:** Stash in metadata dict.

### Multiple gl.call sites / multiple .cu files
| Option | Description | Selected |
|--------|-------------|----------|
| Dict keyed by src_path in metadata | Lazily create per distinct .cu, reuse across sites | ✓ |
| Single .cu per kernel (restrict) | Error on multiple distinct .cu files | |
| You decide / describe | — | |

**User's choice:** Dict keyed by src_path in metadata (matches existing by_libpath grouping).

### Proving no redundant parse (SC3)
| Option | Description | Selected |
|--------|-------------|----------|
| Parse counter + assertion | Count parses/CompilerInstance creations, assert == distinct .cu count | ✓ |
| Document only, no counter | Prose + eyeballed timing | |
| You decide / describe | — | |

**User's choice:** Parse counter + assertion.

---

## f64/fp64 Handling (BUG-02)

### Policy
| Option | Description | Selected |
|--------|-------------|----------|
| Raise clear error | Reject f64/fp64 at the gl.call boundary; full Fp64 stays out of scope | ✓ |
| Keep coercion, warn + document | Keep Fp32 downcast but warn + document | |
| You decide / describe | — | |

**User's choice:** Raise a clear error.

### Where the guard lives
| Option | Description | Selected |
|--------|-------------|----------|
| In call_extern (semantic), + drop map rows | Early dtype check in call_extern + remove f64 rows | |
| In backend dtype_to_scalar (compiler.py) | Backend map raises at spec-building time | |
| Both layers (defense in depth) | Early friendly check + backend backstop | ✓ |
| You decide / describe | — | |

**User's choice:** Both layers.
**Notes:** The semantic-layer check is a pure dtype-string check (no CUDA import), preserving frontend/backend layering. BUG-01 (dead code compiler.py:510-513) is a mechanical delete — no decision.

---

## Graceful Degradation (INFER-06)

### Absent-hook behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Raise 'requires CUDA backend' | Clean error; gl.call is inherently CUDA-specific | ✓ |
| Fall back to first-input inference | Silent degradation preserving old path | |
| You decide / describe | — | |

**User's choice:** Raise "gl.call() extern CUDA calls require the CUDA backend".

### When the raise activates
| Option | Description | Selected |
|--------|-------------|----------|
| Raise activates in Phase 2 | Phase 1 only exposes the hook; call_extern unchanged, 4 tests pass | ✓ |
| Add guard now in Phase 1 | Defensive check in Phase 1 even though hook unused | |
| You decide / describe | — | |

**User's choice:** Activates in Phase 2.
**Notes:** Reinforces the Phase 1 scope line — build the pipe, don't run water through it.

---

## the agent's Discretion

- Final `codegen_fns` key name for the hook.
- Exact wording of the f64 and absent-hook error messages.
- Precise construction site of the inference object in `get_codegen_implementation` and how sm/resource_dir/includes thread in.
- Precise coroutine suspend/resume mechanism over the existing `ExecutionContext`/`SwitchTo`, provided the flow stays parked before `HandleTranslationUnit` completes.

## Deferred Ideas

- AUTO-01 — optional/auto-derived `result_layout` (v2).
- FP64-01 — full Fp64 pipeline support.
- Cross-kernel reuse of the live parse (rejected for Phase 1; AST is consumed by codegen).
- Coroutine/ABI hardening (x86-64-only ABI, dangling captures, `__builtin_unreachable`) — only if suspend/resume forces it.
