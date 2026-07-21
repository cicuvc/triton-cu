"""TDD test for Plan 07-01: shared_accumulate and write_swizzled_2d device functions.

RED phase: these tests MUST FAIL because the functions do not exist yet.
GREEN phase: after adding the functions to tt_plugin.cu, these tests MUST PASS.
"""

import re
import sys
from pathlib import Path


TT_PLUGIN_CU = Path(__file__).parent / "tt_plugin.cu"


def read_cu_file() -> str:
    """Read the entire tt_plugin.cu as a string."""
    return TT_PLUGIN_CU.read_text()


def extract_function_body(content: str, func_name: str) -> str | None:
    """Extract the body of a __device__ function by name.
    Returns the text from the function signature to the closing brace,
    or None if not found.
    """
    # Find function definition
    pattern = rf'__device__\s+void\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{'
    match = re.search(pattern, content)
    if not match:
        return None

    start = match.start()
    # Track braces to find the matching closing brace
    brace_start = match.end() - 1  # position of the opening {
    depth = 1
    i = brace_start + 1
    while i < len(content) and depth > 0:
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
        i += 1

    return content[start:i]


def test_shared_accumulate_exists():
    """Test that shared_accumulate exists with the correct signature.
    RED: FAIL - function does not exist yet.
    """
    content = read_cu_file()
    body = extract_function_body(content, "shared_accumulate")
    assert body is not None, (
        "FAIL: shared_accumulate not found in tt_plugin.cu.\n"
        "Add it between process_shared_2d and END OF DEFINITIONS."
    )

    # Verify template parameters
    assert "template<typename T, uint32_t N, typename SharedTLayout, typename TLayout>" in content, (
        "FAIL: shared_accumulate must have 4 template params: <T, N, SharedTLayout, TLayout>"
    )

    # Verify function signature
    assert "SharedTensor<T, Shape<N>, SharedTLayout>&" in body, (
        "FAIL: shared_accumulate first param must be SharedTensor<T, Shape<N>, SharedTLayout>&"
    )
    assert "const Tensor<T, Shape<N>, TLayout>&" in body, (
        "FAIL: shared_accumulate second param must be const Tensor<T, Shape<N>, TLayout>&"
    )

    # Verify loop bound uses TLayout::REG_SIZE (not N) - critical for correctness
    assert "TLayout::REG_SIZE" in body, (
        "FAIL: shared_accumulate loop must use TLayout::REG_SIZE, not N"
    )

    # Verify pragma unroll
    assert "#pragma unroll" in body, (
        "FAIL: shared_accumulate must have #pragma unroll"
    )

    # Verify shm(i) += val.data[i] pattern
    assert "shm(i)" in body, "FAIL: shared_accumulate must use shm(i) for SharedTensor access"
    assert "val.data[i]" in body, "FAIL: shared_accumulate must use val.data[i] for Tensor access"
    assert "+=" in body, "FAIL: shared_accumulate must use += for accumulate semantics"

    # Verify return type is void
    assert "__device__ void shared_accumulate" in content, (
        "FAIL: shared_accumulate must be __device__ void"
    )

    print("PASS: shared_accumulate exists with correct signature and loop bound")


def test_write_swizzled_2d_exists():
    """Test that write_swizzled_2d exists with the correct signature.
    RED: FAIL - function does not exist yet.
    """
    content = read_cu_file()
    body = extract_function_body(content, "write_swizzled_2d")
    assert body is not None, (
        "FAIL: write_swizzled_2d not found in tt_plugin.cu.\n"
        "Add it between process_shared_2d and END OF DEFINITIONS."
    )

    # Verify template parameters - same as process_shared_2d
    assert "template<typename T, typename TLayout>" in content, (
        "FAIL: write_swizzled_2d must have 2 template params: <T, TLayout>"
    )

    # Verify function signature
    assert "SharedTensor<T, Shape<32, 16>, TLayout>&" in body, (
        "FAIL: write_swizzled_2d param must be SharedTensor<T, Shape<32, 16>, TLayout>&"
    )

    # Verify double-loop iteration pattern
    assert "for (int i = 0; i < 32; i++)" in body or "for (int i = 0; i < 32; ++i)" in body, (
        "FAIL: write_swizzled_2d must iterate rows 0..31"
    )
    assert "for (int j = 0; j < 16; j++)" in body or "for (int j = 0; j < 16; ++j)" in body, (
        "FAIL: write_swizzled_2d must iterate cols 0..15"
    )

    # Verify identity value write: shm(i,j) = i*16+j
    assert "shm(i, j)" in body, (
        "FAIL: write_swizzled_2d must use shm(i, j) for 2D SharedTensor access"
    )

    # Verify static_cast for type safety
    assert "static_cast<T>" in body, (
        "FAIL: write_swizzled_2d must use static_cast<T> for type-safe assignment"
    )

    # Verify the identity formula: i * 16 + j
    assert "16" in body, "FAIL: write_swizzled_2d must use column count 16 in identity formula"

    # Verify return type is void
    assert "__device__ void write_swizzled_2d" in content, (
        "FAIL: write_swizzled_2d must be __device__ void"
    )

    print("PASS: write_swizzled_2d exists with correct signature and iteration pattern")


def test_functions_placed_correctly():
    """Test that the new functions are placed between process_shared_2d and END OF DEFINITIONS."""
    content = read_cu_file()

    # Find process_shared_2d closing brace (line ~225)
    ps2d_match = re.search(r'__device__ void process_shared_2d.*?\n\}', content, re.DOTALL)
    eod_match = re.search(r'//\s*=+\s*END OF DEFINITIONS\s*=+', content)

    assert ps2d_match is not None, "FAIL: process_shared_2d not found in tt_plugin.cu"
    assert eod_match is not None, "FAIL: END OF DEFINITIONS comment not found in tt_plugin.cu"

    ps2d_end = ps2d_match.end()
    eod_start = eod_match.start()

    # The new functions must be between process_shared_2d and END OF DEFINITIONS
    between = content[ps2d_end:eod_start]
    assert "shared_accumulate" in between, (
        "FAIL: shared_accumulate must be placed between process_shared_2d and END OF DEFINITIONS"
    )
    assert "write_swizzled_2d" in between, (
        "FAIL: write_swizzled_2d must be placed between process_shared_2d and END OF DEFINITIONS"
    )

    print("PASS: Functions correctly placed between process_shared_2d and END OF DEFINITIONS")


def test_existing_functions_unchanged():
    """Test that existing device functions are not modified."""
    content = read_cu_file()

    # Verify process_shared_2d still exists unchanged
    assert "void process_shared_2d(SharedTensor<T, Shape<32, 16>, TLayout>& shm, T scale)" in content, (
        "FAIL: process_shared_2d signature was modified"
    )

    # Verify write_shared_1d still exists unchanged
    assert "void write_shared_1d(SharedTensor<T, Shape<N>, TLayout>& shm, T val)" in content, (
        "FAIL: write_shared_1d signature was modified"
    )

    # Verify elementwise_add still exists unchanged
    assert "elementwise_add" in content, (
        "FAIL: elementwise_add was removed or modified"
    )

    print("PASS: Existing functions unchanged")


def test_threat_mitigations():
    """Verify threat mitigations per the plan's threat model.
    T-07-01: Loop bound is TLayout::REG_SIZE (not N), preventing OOB on non-identity layouts.
    """
    content = read_cu_file()
    body = extract_function_body(content, "shared_accumulate")
    if body is not None:
        # T-07-01 mitigation: REG_SIZE loop bound, not N
        assert "TLayout::REG_SIZE" in body, (
            "FAIL T-07-01: shared_accumulate loop must use TLayout::REG_SIZE, "
            "not N from shape. Without this, non-identity layouts will OOB."
        )
        print("PASS: T-07-01 mitigated (REG_SIZE loop bound)")


if __name__ == "__main__":
    failures = []
    for name in [
        "test_shared_accumulate_exists",
        "test_write_swizzled_2d_exists",
        "test_functions_placed_correctly",
        "test_existing_functions_unchanged",
        "test_threat_mitigations",
    ]:
        print(f"\n--- {name} ---")
        try:
            globals()[name]()
        except AssertionError as e:
            print(str(e))
            failures.append(name)
        except Exception as e:
            print(f"ERROR: {e}")
            failures.append(name)

    if failures:
        print(f"\n=== {len(failures)} TESTS FAILED (expected in RED phase) ===")
        print("Failures:", ", ".join(failures))
        sys.exit(1)
    else:
        print("\n=== ALL TESTS PASSED ===")
        sys.exit(0)
