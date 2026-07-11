# Phase 1: Seam & Cleanup - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a clean way for the backend-agnostic Gluon semantic layer to invoke CUDA return-type inference (via the backend `codegen_fns` hook), guarantee the `.cu` source is not parsed twice across the semantic and `llir` stages, and clear the two bundled bugs — **before** touching the actual inference data flow (which is Phase 2).

**In scope:** Expose the inference hook, add the cached/suspended CUDACompiler machinery, single-parse plumbing + verification, f64/fp64 guard, dead-code removal.

**Out of scope (this phase):** Actually consuming the hook in `call_extern` to build result types from CUDA-inferred shape/dtype/layout (Phase 2), layout reconciliation / `convert_layout` (Phase 2), new tests (Phase 3), full Fp64 support (deferred FP64-01), auto-derived `result_layout` (deferred AUTO-01).

**Critical constraint:** `call_extern` behavior is UNCHANGED in Phase 1 — it still infers result types from `first_input` — so all 4 existing extern-call tests pass unchanged (SC5).

</domain>

<decisions>
## Implementation Decisions

### Inference Hook Contract (INFER-06)
- **D-01:** The hook is a **stateful object with methods**, not a bare callable, exposed through `codegen_fns` (from `CUDABackend.get_codegen_implementation`, `compiler.py:246`) — consumed by the semantic layer as `self.builder.codegen_fns[...]`, mirroring `convert_custom_types` / `min_dot_size` (`semantic.py:837,1479`).
- **D-02:** Two methods:
  - `infer_result(func, arg_params, use_fast_math) -> result TensorParameter(s)` — called at **semantic time** (IR-build, in `call_extern`) to get the CUDA-inferred return type. (Consumption of its result is Phase 2; Phase 1 only makes it callable.)
  - `compile_bitcode(requests) -> (bitcode, mangled_names, extractor_names)` — called at the **llir stage** (`_pre_compile_extern_calls`) to emit device bitcode.
  - Both methods are backed by the **same live clang parse** of the `.cu`.
- **D-03 (key layering decision):** Cache the **whole live `CUDACompiler`** with its **coroutine suspended inside `HandleTranslationUnit`**, so the `ASTContext` remains alive and return-type inference (`EvaluateFunctionReturnType` / `TypeInspector`) stays valid. Rationale (user): once the `ASTConsumer` flow completes, clang tears down the `ASTContext` — cached bare `FunctionDecl`s dangle and can no longer support type deduction/inspection. Parking the flow *before* `HandleTranslationUnit` finishes, until LLVM IR generation actually begins, is the safest. This also resolves the "inference needs a parse but `LLVMContext`/sm/resource_dir live in the llir stage" problem: the suspended coroutine carries the parse forward to where the `LLVMContext` exists.
- **D-04:** Coroutine lifetime is **per-compile**: created + suspended at semantic time (AST live for inference), **resumed and consumed** at that same compile's `llir` stage to emit bitcode, then torn down. No cross-kernel reuse of the *live* coroutine; the existing disk cache (`.cu` path in cache key, `_runtime.py`) still handles repeated compiles of the same kernel.

### Single-Parse Strategy (INFER-07)
- **D-05:** The suspended `CUDACompiler` travels from the semantic stage to the `llir` stage by being **stashed in the compile's `metadata` dict** (alongside the existing `extern_call_bitcodes` / `extern_call_mangled` entries). `make_ir` (semantic) creates+suspends+stashes; `_pre_compile_extern_calls` (llir) pulls it out and resumes.
- **D-06:** Multiple `gl.call` sites / multiple `.cu` files: store a **dict keyed by resolved `src_path` → suspended `CUDACompiler`** in metadata. Lazily create one per distinct `.cu`, reuse across call sites to the same file, separate entries for different files. This matches the existing `by_libpath` grouping in `_pre_compile_extern_calls`.
- **D-07:** Prove "no redundant parse" (roadmap SC3) with a **parse counter + assertion**: count actual clang parses / `CompilerInstance` creations per compile and assert it equals the number of distinct `.cu` files. Hard regression guard, not just prose.

### f64/fp64 Handling (BUG-02)
- **D-08:** **Raise a clear error** at the `gl.call` boundary when any arg (or requested result) is `f64`/`fp64` — e.g. "gl.call() does not support float64; full Fp64 support is out of scope (see FP64-01)". No silent `Fp32` truncation.
- **D-09:** **Both layers (defense in depth):**
  - Friendly early dtype check in `call_extern` (`_semantic.py`) — a plain dtype-string check (backend-agnostic, no CUDA import, preserves layering) that raises before building IR or invoking inference, giving a stack trace pointing at the user's `gl.call`.
  - Replace the `"f64"→Fp32` / `"fp64"→Fp32` rows in `dtype_to_scalar` (`compiler.py:542`) with an explicit error as a backend backstop.

### Dead Code (BUG-01)
- **D-10:** Mechanical — delete the unreachable duplicate `return ret` block at `compiler.py:510-513`. No decision required.

### Graceful Degradation (INFER-06)
- **D-11:** When the inference hook key is absent from `codegen_fns` (interpreter, AMD, any non-CUDA backend), the eventual behavior is to **raise a clear error**: "gl.call() extern CUDA calls require the CUDA backend." `gl.call` is inherently CUDA-specific (loads `.cu`, clang CodeGen) — there is no meaningful non-CUDA behavior. A clean raise satisfies roadmap SC2 ("no crash when the hook is absent").
- **D-12 (timing):** The absent-hook raise **activates in Phase 2**, when `call_extern` actually consumes the hook. In **Phase 1**, `call_extern` is untouched (still first-input inference) and nothing consumes the hook, so "graceful when absent" holds by construction and the 4 tests pass unchanged.

### the agent's Discretion
- Exact `codegen_fns` key name for the hook (suggested working name: `infer_extern_call_result` or similar) — planner/executor may choose the final name.
- Exact error-message wording for f64 and absent-hook cases (intent is locked; phrasing is flexible).
- Precise construction site of the CUDACompiler object within `get_codegen_implementation` and how `sm`/`resource_dir`/include paths are threaded into it.
- Precise mechanism for suspending/resuming the coroutine at the stage boundaries, given the existing `ExecutionContext`/`SwitchTo` machinery — provided it keeps the flow parked before `HandleTranslationUnit` completes.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference implementation (full inference pipeline PoC)
- `/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.cpp` — standalone proof-of-concept with `TypeBuilder`, `TypeInspector`, `FunctionResolver`, `CUDACompiler::EvaluateFunctionReturnType()`. The abstractions to integrate for the hook.
- `/home/cicuvc/cs/project/nks/lab/cu_compiler_v2.h` — headers/struct definitions for the above.

### The seam (frontend ↔ backend)
- `third_party/nvidia/backend/compiler.py` — `get_codegen_implementation` (line 246, add the hook here), `_pre_compile_extern_calls` (line 515, resumes the coroutine to emit bitcode), `dtype_to_scalar` (line 538-545, f64 rows to replace), dead code (lines 510-513), CUDA compile call (line 583).
- `python/triton/compiler/compiler.py` — `codegen_fns = backend.get_codegen_implementation(options)` (line 304) → `src.make_ir(...)` (line 307); how the hook threads through.
- `python/triton/compiler/code_generator.py` — `ast_to_ttir` (line 1662), `self.builder.codegen_fns = codegen_fns` (line 312); the propagation into the builder.
- `python/triton/experimental/gluon/_runtime.py` — `make_ir` (line 47), `ast_to_ttir` call (line 67); Gluon's IR entry that receives `codegen_fns`; also the `.cu`-in-cache-key logic.
- `python/triton/experimental/gluon/language/_semantic.py` — `call_extern` (line 250-273); where the f64 early guard lives and where the hook would be invoked (Phase 2). **Unchanged in Phase 1 except the f64 guard.**
- `python/triton/experimental/gluon/language/_core.py` — `gl.call` user API (line 774).

### CUDA compiler / bindings
- `python/src/clang_compiler.cc` / `.h` — `CUDACompiler` coroutine, `ExecutionContext`/`SwitchTo`, `CustomAstConsumer::HandleTranslationUnit`, `EvaluateFunctionReturnType`, `TypeInspector`, `FunctionResolver`, `linkBitcodeToModule`.
- `python/src/llvm.cc` — Python bindings: `ScalarType`, `TensorParameter`, `CudaFuncRequest`, `compile_cuda_to_module`, `extract_extern_call_specs`, `link_cuda_bitcode`.

### Project docs
- `.planning/PROJECT.md` — locked decisions (result_layout required, infer-at-semantic-time, codegen_fns seam).
- `.planning/REQUIREMENTS.md` — INFER-06, INFER-07, BUG-01, BUG-02 (Phase 1 scope); TEST-* (Phase 3).
- `.planning/codebase/CONCERNS.md` — fragility of `linkBitcodeToModule` (§Fragile Areas), coroutine dispatcher fragility (stack-dangling `[&]` captures, `__builtin_unreachable`, X64-only ABI) — relevant to the suspend/resume approach.
- `AGENTS.md` — build instructions, extern-call pipeline overview, return-type inference integration plan.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`codegen_fns` dict mechanism** — already carries `convert_custom_types` and `min_dot_size` from backend → builder → semantic layer. The inference hook slots in the same way; no new plumbing pattern needed.
- **`metadata` dict** — already threads `extern_call_bitcodes`, `extern_call_mangled`, `extern_call_extractor_names` across stages; the suspended CUDACompiler dict rides the same channel.
- **`by_libpath` grouping** — `_pre_compile_extern_calls` already groups specs per `.cu`; the per-source suspended-compiler dict aligns with it.
- **Existing C++ inference machinery** — `TypeInspector`, `FunctionResolver`, `EvaluateFunctionReturnType` are already implemented (`clang_compiler.cc`); Phase 1 exposes/plumbs them, it does not rewrite them.
- **`ScalarType` / `TensorParameter` / `CudaFuncRequest` bindings** — already exist in `llvm.cc` for building requests and reading results.

### Established Patterns
- Backend-agnostic frontend rule: the Gluon semantic layer must NOT import NVIDIA backend code; all CUDA specifics arrive via `codegen_fns`. The f64 early check must therefore be a pure dtype-string check.
- Each compile stage is a `compile_ir(mod, metadata) -> mod` callable; state passed between stages lives in `metadata`.
- Coroutine dispatch: every `CUDACompiler` method pushes a `[&]`-capturing task to `TaskQueue` then `SwitchTo` the clang thread. The suspend/resume design must respect this protocol (captures must outlive the switch) — see CONCERNS.md fragility notes.

### Integration Points
- `get_codegen_implementation` (`compiler.py:246`) — construct/expose the inference object here.
- `make_ir` (Gluon `_runtime.py:47` → `ast_to_ttir`) — semantic stage where the object is created/suspended and stashed in metadata.
- `_pre_compile_extern_calls` (`compiler.py:515`) — llir stage where the object is pulled from metadata and resumed to emit bitcode.
- `call_extern` (`_semantic.py:250`) — the f64 early guard; the (Phase-2) hook invocation point.

</code_context>

<specifics>
## Specific Ideas

- User's explicit reasoning on caching the whole CUDACompiler (verbatim intent): the coroutine approach is *more* stable than caching parsed decls, because after the `ASTConsumer` flow completes clang's AST is no longer used, internal pointers are invalidated, and the `ASTContext` is torn down — cached `FunctionDecl`s would be invalid for further type deduction/inspection. Keeping the flow parked before `HandleTranslationUnit` ends, until LLVM IR generation truly begins, is the safest.
- Prefer failing loud over silent coercion (drove both the f64 error and the absent-hook error decisions).
- Phase 1 is deliberately a "build the pipe, don't run water through it" phase — the sharp scope line at "call_extern unchanged / 4 tests pass" is intentional.

</specifics>

<deferred>
## Deferred Ideas

- **AUTO-01** — Make `result_layout` optional / auto-derived from CUDA-inferred layout. Deferred (v2); user keeps `result_layout` explicit.
- **FP64-01** — Full `Fp64` support through the pipeline (`ScalarType::Fp64`, `getQualTypeFromScalarType`, lowering, `Ctx.DoubleTy`). Deferred; Phase 1 only adds the guard/error.
- **Cross-kernel reuse of the live parse** — a persistent process-global cache of parsed `.cu` across kernels was considered but rejected for Phase 1 because codegen consumes the AST; disk cache covers repeated compiles. Could revisit if parse cost is proven a bottleneck.
- **Coroutine/ABI hardening** (x86-64-only `X64SysVABI`, stack-dangling captures, `__builtin_unreachable`) — out of scope; touch only if the suspend/resume work forces it.

None of these belong in Phase 1.

</deferred>

---

*Phase: 1-seam-cleanup*
*Context gathered: 2026-07-11*
