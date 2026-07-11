---
status: testing
phase: 02-semantic-time-inference
source: [02-VERIFICATION.md]
started: 2026-07-11T00:00:00Z
updated: 2026-07-11T00:00:00Z
---

## Current Test

number: 1
name: Build libtriton.so and deploy for local dev
expected: |
  `bash build.sh` completes successfully and `build/libtriton.so` is produced,
  then copied to `python/triton/_C/libtriton.so`.
awaiting: user response

## Tests

### 1. Build succeeds
expected: |
  `bash build.sh && cp build/libtriton.so python/triton/_C/libtriton.so`
  completes with no errors (uses self-compiled LLVM + clang per AGENTS.md).
result: [pending]

### 2. Full extern-call suite passes (5 tests)
expected: |
  `PYTHONPATH="python:third_party/nvidia" pytest python/test/gluon/test_extern_call.py -v -s`
  → 5 passed (4 original + new test_gl_call_no_inference_hook_raises).
result: [pending]

### 3. INFER-01/02: fixed-layout reduce uses CUDA inference (not first_input fallback)
expected: |
  test_reduce_different_shape passes with the return dtype+shape obtained from
  CUDA inference via LookupFunctionWithPlaceholderFallback — confirming Gap 1 is
  closed at runtime, not just at source level.
result: [pending]

### 4. INFER-03: inference failures propagate (no silent swallow)
expected: |
  The try/except RuntimeError wrapper is gone; a genuine CUDA inference failure
  surfaces as an error rather than silently falling back to first_input shape.
result: [pending]

### 5. INFER-06: hook-absent raise fires at runtime
expected: |
  test_gl_call_no_inference_hook_raises passes: with the inference hook stripped,
  gl.call() raises with message "gl.call() extern CUDA calls require the CUDA
  backend. No inference hook (infer_extern_call_result) found in codegen_fns."
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
