# Phase 5: MLIR Op Relaxation + Spec Extraction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-15
**Phase:** 05-MLIR Op Relaxation + Spec Extraction
**Areas discussed:** ODS relaxation strategy, SpecInput data model for shared memory, Lit test design

---

## ODS relaxation strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Variadic<AnyTypeOf<[Tensor,MemDesc]>> | Single mixed input list. MLIR-idiomatic, but downstream passes that assume tensor-only inputs via cast<RankedTensorType> will break at runtime | ✓ |
| Separate Variadic params (tensor+memdesc) | Two separate input lists: tensor operands + memdesc operands. Preserves tensor-only assumption in downstream passes but splits arg ordering | |

**User's choice:** Variadic<AnyTypeOf<[TT_Tensor, TTG_MemDescType]>>
**Notes:** Chosen as the simplest change and most natural for argument ordering. Downstream pass blast radius will be verified during planning.

---

## SpecInput data model for shared memory

| Option | Description | Selected |
|--------|-------------|----------|
| Add a memory_space discriminator | Single struct with `memory_space` field ('register'/'shared') + `std::optional<>` shared-specific fields. Non-shared inputs default to 'register' with empty optional fields | |
| Variant-based (separate types) | Replace SpecInput with `std::variant<TensorSpecInput, SharedSpecInput>`. Clean separation at the type level. Requires variant visitors everywhere SpecInput is consumed | ✓ |

**User's choice:** Variant-based — `std::variant<TensorSpecInput, SharedSpecInput>`
**Notes:** Clean type-level separation chosen over optional/sentinel fields. Both variants remain within the anonymous namespace in `clang_compiler.cc`.

---

## Lit test design

| Option | Description | Selected |
|--------|-------------|----------|
| Two tests: mixed + tensor-only parse verification | One lit test parses a ttg.extern_call with mixed operand types (memdesc + tensor) and checks --verify-diagnostics. One test for tensor-only regression confirmation | ✓ |
| Three tests: mixed, tensor-only, + extraction smoke | Add a third test that exercises extractExternCallSpecs by checking output JSON — needs the extraction path fully wired | |

**User's choice:** Two tests — mixed parse verification + tensor-only regression
**Notes:** Extraction smoke testing deferred to Phase 6 when the CUDA compiler consumes the specs end-to-end.

---

## the agent's Discretion

No areas deferred to the agent — all three gray areas were decided explicitly by the user.

## Deferred Ideas

None — discussion stayed within phase scope.
