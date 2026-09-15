---
name: ort-test
description: Run ONNX Runtime tests. Use this skill when asked to run tests, debug test failures, or find and execute specific test cases in ONNX Runtime.
---

# Running ONNX Runtime Tests

ONNX Runtime uses **Google Test** for C++ and **unittest** (preferred) / **pytest** for Python.

## C++ tests

### Test executables

| Executable | What it tests |
|---|---|
| `onnxruntime_test_all` | Core framework, graph, optimizer, session tests |
| `onnxruntime_provider_test` | Operator/kernel tests (Conv, MatMul, etc.) across execution providers |

### Two `attention_op_test.cc` files — don't confuse them

There are two same-named files testing **different operators**. Both build into
`onnxruntime_provider_test`:

| Path | Operator | gtest suite |
|---|---|---|
| `test/providers/cpu/llm/attention_op_test.cc` | **ONNX-domain** `Attention` (opset 23/24) | `AttentionTest.*` |
| `test/contrib_ops/attention_op_test.cc` | **contrib** MultiHeadAttention / GroupQueryAttention | `ContribOpAttentionTest.*` |

The MEA negative-offset regression tests (`Attention_Causal_NonPadKVSeqLen_MEA_*`,
e.g. `..._MEA_NegOffset_ForceFlashDisabled_FP16_CUDA`) live in the **providers/cpu/llm** file —
the ONNX-domain op.

Use `--gtest_filter` to select specific tests:

```bash
./onnxruntime_provider_test --gtest_filter="*Conv3D*"
```

### Running tests

**Always run from the build output directory** — tests may fail to find dependencies otherwise.

```bash
# Linux
cd build/Linux/Release
./onnxruntime_provider_test --gtest_filter="*TestName*"

# macOS
cd build/MacOS/Release
./onnxruntime_provider_test --gtest_filter="*TestName*"

# Windows
cd build\Windows\Release
.\onnxruntime_provider_test.exe --gtest_filter="*TestName*"
```

You can also run all tests via the build script (assumes a prior successful build):

```bash
./build.sh --config Release --test
.\build.bat --config Release --test    # Windows
```

### Locating the build output directory

The default path follows the pattern `build/<Platform>/<Config>/` where Platform is `Linux`, `MacOS`, or `Windows`. With Visual Studio multi-config generators on Windows, the config may appear twice (e.g., `build/Windows/Release/Release/`). The path can also be customized via `--build_dir`.

If you can't find a test binary, search for it:

```powershell
# Windows
Get-ChildItem -Path build -Recurse -Filter "onnxruntime_provider_test.exe" | Select-Object -ExpandProperty FullName

# Linux/macOS
find build -name "onnxruntime_provider_test" -type f
```

## Python tests

Use `pytest` as the test runner:

```bash
pytest onnxruntime/test/python/test_specific.py                          # entire file
pytest onnxruntime/test/python/test_specific.py::TestClass::test_method  # specific test
pytest -k "test_keyword" onnxruntime/test/python/                        # by keyword
```

Python test naming convention: `test_<method>_<expected_behavior>_[when_<condition>]`

## Agent tips

- **Activate a Python virtual environment** before running tests. See "Python > Virtual environment" in `AGENTS.md`.
- **Beware false-green results** — a green run does not always prove anything. See the
  "False-green taxonomy" section below for the four ways a test can pass without testing
  your change.
- **Redirect test output to a file** (e.g., `> test_output.txt 2>&1`) — output can be large.
- For C++ tests, verify the build directory exists and a prior build completed before running.
- Use `--gtest_filter` to run a targeted subset when the full suite takes too long.
- **Running WebGPU tests locally on Linux without a GPU** — WebGPU op tests build into `onnxruntime_provider_test` and can run against a software Vulkan adapter (Mesa lavapipe). See the `webgpu-local-testing` skill.

## False-green taxonomy — ways a test can "pass" without proving anything

A green result is not always a real pass. Watch for all five modes:

1. **Zero-match filter.** A `--gtest_filter` that matches no tests still exits 0 (green).
   Confirm the `[==========] N tests ran` line is non-zero — a zero-match run prints
   `0 tests from 0 test suites`. Many operator/kernel gtests run only in
   **`onnxruntime_provider_test`** (CI runs this), NOT `onnxruntime_test_all`; the wrong
   binary matches nothing and looks green.
2. **Stale binary from an incremental build.** If the build did not actually recompile your
   change (e.g. a header not tracked by the compiler's depfile), the "passing" run executes
   the OLD code. A test that was failing cannot truly flip to passing without a real
   rebuild — treat an unexpected FAIL→PASS with suspicion and confirm the linked artifact's
   mtime advanced. CUDA/CUTLASS instance (nvcc depfiles don't track `cutlass_fmha/*.h`): see
   the `cuda-cutlass-fmha-incremental-rebuild` skill.
3. **Checking the wrong artifact's freshness.** With a dlopen'd shared provider (e.g.
   `libonnxruntime_providers_cuda.so`), the test executable is NOT relinked when the provider
   recompiles — its mtime stays old while the `.so` advances. Verify the artifact that
   actually links your change, not the test exe. Detail: `cuda-cutlass-fmha-incremental-rebuild`
   skill.
4. **A correct fallback path masks the intended path.** A value-only assertion can pass via a
   *different, correct* code path without ever exercising the one you meant to test (e.g. a
   test meant for MEA silently handled by the unfused fallback). Assert/verify **which path
   ran**, not just the output value — see "Verify which path/kernel actually executed" below.
5. **Arch-portability false-green (verified on only one GPU arch).** A CUDA kernel that
   launches on a large-dynamic-smem arch (e.g. sm90/H100, ~227KB) can **fail to launch** on a
   smaller opt-in cap (sm86/89 ~99KB, sm80 ~163KB) with `CUDA failure 1: invalid argument` —
   and a path with no fallback (e.g. ORT's MEA) turns that into a hard error, not a silent
   degrade. So a green run on your local GPU can mask a launch failure on CI's arch. Verify
   arch-portability, or pick a config whose shared-memory footprint fits **every** target arch
   (e.g. a small `head_size`). Concrete instance: CUTLASS MEA `head_size=512` FP16 exceeds
   sm86's smem opt-in cap and dies at launch — live bug #28388 (the
   `cuda-attention-kernel-patterns` skill §1 has the dispatch detail).

## Verify which path/kernel actually executed

Value equality alone does not prove the intended code path ran — a correct fallback can
produce the right answer (false-green mode 4 above). When a test targets a specific
kernel/path, confirm it actually dispatched there instead of trusting the output:

- Enable verbose logging and check the dispatch log line. ORT attention logs one of these
  exact strings (`core/providers/cuda/llm/attention.cc`):
  - `ONNX Attention: using Flash Attention` (:1400)
  - `ONNX Attention: using Memory Efficient Attention` (:1451)
  - `Attention: using unified unfused path` (:1482) — note: **no `ONNX ` prefix** and it
    reads "unified unfused path", not "Unfused".
- Or force the path via the relevant env var / build config AND add a compile-time guard so
  the test **SKIPs** (not silently passes) when the target path is unavailable — e.g.
  `SKIP_IF_MEA_NOT_COMPILED`.

Operator-specific routing/forcing details: `cuda-attention-kernel-patterns` skill §1/§7.
