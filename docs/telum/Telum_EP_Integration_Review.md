# Telum Execution Provider (s390x/z16) Integration Review

Date: 2026-02-10

Repo: `onnxruntime/` (local workspace)

Branch reviewed (local): `feature/telum-ep-e2e-integration` (dirty working tree at review time)

This document is intentionally verbose. If you want the actionable checklist, see `docs/telum/Telum_EP_TODO.md`.

## Goal And Definition Of "Full Telum Integration"

This effort is adding a first-class, in-tree **Telum Execution Provider (EP)** for **IBM Z s390x (z16+)** that uses:

- **zDNN** user-space library
- **NNPA (Neural Network Processing Assist)** hardware on Telum

For the purposes of this repo, "full Telum integration" means:

1. **Build-time**
   - `onnxruntime_USE_TELUM` toggles the EP cleanly.
   - Non-s390x builds remain unaffected (Telum OFF by default, no new deps).
   - s390x builds fail fast with clear messages when zDNN is missing.
2. **Runtime / selection**
   - Provider appears in `GetAvailableExecutionProviderNames()` only when built (`USE_TELUM`).
   - Provider can be selected from:
     - Python provider list (`providers=[('TelumExecutionProvider', {...}), ...]`)
     - Generic C API `OrtApis::SessionOptionsAppendExecutionProvider(...)` string-based selection
3. **Graph partitioning**
   - Telum EP claims only nodes it can actually execute, based on:
     - static shapes
     - supported types
     - supported per-op shape/broadcast/attribute patterns
     - kernel availability
   - Rejection reasons are diagnosable (logs and/or explicit errors when CPU fallback is disabled).
4. **Correctness**
   - Implemented kernels match ONNX semantics for the supported subspace.
   - Unsupported semantics are rejected at capability time when possible.
5. **Tests**
   - There are real tests that:
     - explicitly select Telum
     - **disable CPU EP fallback** so the test fails if it silently ran on CPU
     - are skipped on hosts without NNPA/zDNN
6. **Performance (phase-dependent)**
   - A "correct-but-slow" implementation is acceptable initially.
   - Production viability requires caching/prepacking/transformation reuse of constant weights.

This review focuses on the state of the integration so far, risks, and the path to meeting the above definition.

## Key Constraints And What They Mean In Practice

### Platform / environment

- Telum EP is **s390x-only** and **requires z16+** (NNPA).
- zDNN must be installed and discoverable via `ZDNN_ROOT`.
- This workspace host is macOS; Telum builds/tests cannot run locally here.
  - This makes compile gating and CI/remote s390x validation particularly important.

### zDNN behavioral constraints that matter to ONNX Runtime

The implementation already reflects a few important zDNN realities:

- **Static shapes**: zDNN generally expects fixed shapes/descriptors.
- **MatMul bias parameter is required**: `zdnn_matmul_op` expects a non-null `input_c` (bias vector/matrix), even when you conceptually want "no bias". This requires providing an explicit zero bias tensor.
- **Elementwise ops do not broadcast**: zDNN `zdnn_add/sub/mul/...` operate on same-shaped tensors. Any ONNX-style broadcasting must be handled by the caller (EP) via:
  - graph fusions (e.g., MatMul+Add -> Gemm) where possible, and/or
  - explicit broadcast expansion or CPU fallback inside the Telum kernel for limited patterns.

## What Exists In-Tree Today (Summary)

### Build integration (CMake)

State: **Integrated**, behind `onnxruntime_USE_TELUM`, and fails fast on non-s390x.

Relevant files:

- `cmake/CMakeLists.txt`
  - Adds option `onnxruntime_USE_TELUM`.
- `cmake/onnxruntime_providers.cmake`
  - Includes `onnxruntime_providers_telum.cmake` when enabled.
- `cmake/onnxruntime.cmake`
  - Links `onnxruntime_providers_telum` into the internal provider libs list.
- `cmake/onnxruntime_providers_telum.cmake`
  - Validates `CMAKE_SYSTEM_PROCESSOR` is `s390x`.
  - Requires `ZDNN_ROOT` (or env var).
  - Locates `zdnn/zdnn.h` and `libzdnn`.
  - Defines `USE_TELUM`.
  - Builds `onnxruntime_providers_telum` static lib and links zDNN.
  - Adds `-march=z16 -mtune=z16`.

Assessment:

- This is the right overall shape.
- The "fail fast on non-s390x" decision is good: it prevents accidentally sprinkling z16 compiler flags into other builds.

### Provider registration and selection plumbing

State: **Integrated** at the ORT core selection points.

Relevant files:

- `include/onnxruntime/core/graph/constants.h`
  - Adds `kTelumExecutionProvider = "TelumExecutionProvider"`.
- `onnxruntime/core/providers/get_execution_providers.cc`
  - Adds Telum into ordered provider list (available under `USE_TELUM`).
- `onnxruntime/core/providers/telum/telum_provider_factory_creator.{h,cc}`
  - Parses provider options and returns an `IExecutionProviderFactory`.
- `onnxruntime/core/providers/provider_factory_creators.h`
  - Includes Telum factory creator under `USE_TELUM`.
- `onnxruntime/core/session/provider_registration.cc`
  - Adds Telum into generic `OrtApis::SessionOptionsAppendExecutionProvider(...)` string selection list:
    - canonical name: `TelumExecutionProvider`
    - short name: `Telum`
- `onnxruntime/python/onnxruntime_pybind_state.cc`
  - Adds Telum to Python-side EP factory selection under `USE_TELUM`.

Assessment:

- This is the correct direction: a minimal, in-tree EP that can be selected like other built-in EPs.
- One gap is the C++ convenience API: there is no `AppendExecutionProvider_Telum()` wrapper, but the generic append API exists. That is acceptable for now as long as docs reflect reality.

### Kernel registry

State: **Implemented** and shared across sessions.

Relevant files:

- `onnxruntime/core/providers/telum/telum_kernel_registry.{h,cc}`
  - Creates a shared static `KernelRegistry`.
  - Registers kernel create infos for the Telum kernels.
- `onnxruntime/core/providers/telum/telum_execution_provider.cc`
  - `GetKernelRegistry()` returns the shared registry.

Assessment:

- This is required for `GetCapability()` to claim anything and is implemented correctly.
- Registry currently includes:
  - `MatMul`
  - `Gemm`
  - Elementwise: `Add/Sub/Mul/Div/Min/Max`
  - Activations: `Relu` (ONNX), `Softmax` (ONNX), `Gelu` (MS), `Tanh/Sigmoid/Exp/Log/Sqrt`
  - Normalization: `LayerNormalization` (ONNX)

### Implemented kernels and core correctness work

State: **Material correctness improvements** were made, especially around MatMul/Gemm semantics and zDNN constraints.

Relevant files:

- Tensor conversions:
  - `onnxruntime/core/providers/telum/utils/tensor_converter.{h,cc}`
    - Added the ability to use an alternate **logical shape** for zDNN descriptors while keeping ORT tensor shape unchanged.
    - Added raw-data to ztensor conversion helper (used for zero bias creation).
- MatMul:
  - `onnxruntime/core/providers/telum/kernels/math/matmul.cc`
    - Uses `ZDNN_2D` / `ZDNN_3DS` properly.
    - Always supplies a non-null bias tensor `input_c` (zero bias when needed).
    - Supports:
      - unstacked 2D
      - stacked batch (collapsing ONNX batch dims into a single stack dim)
      - full broadcast of a fully-unstacked operand across all batch dims using `zdnn_matmul_bcast_op`
- Gemm:
  - `onnxruntime/core/providers/telum/kernels/math/gemm.cc`
    - Fixes attribute defaults (ONNX defaults, no enforce-on-missing).
    - Uses zDNN matmul (+ transpose op when needed).
    - Fuses bias only for the safe subset; otherwise uses zero bias and applies alpha/beta and broader C broadcasting on CPU post-processing.
- Elementwise:
  - `onnxruntime/core/providers/telum/kernels/math/elementwise.cc`
    - Uses zDNN for same-shape cases (rank <= 4).
    - Implements ONNX-style broadcasting on CPU (rank <= 4) for correctness when zDNN cannot broadcast.
- Activations:
  - `onnxruntime/core/providers/telum/kernels/activation/activation.cc`
    - Straightforward wrappers around zDNN unary ops.

Assessment:

- MatMul and Gemm changes are on the right track: they reflect zDNN API constraints and ONNX semantics.
- The "logical shape" concept is the right abstraction for mapping ONNX broadcast batch dims onto zDNN stack dims.

### Capability gating and crash fixes

State: **Improved** and more honest about what kernels can handle.

Relevant files:

- `onnxruntime/core/providers/telum/telum_execution_provider.cc`
  - Reordered `GetCapability()` to validate shapes and types before op checks.
  - Skips optional null inputs during validation.
  - Adds per-op gating:
    - `Gemm`: requires A/B 2D
    - `MatMul`: requires either fully-stacked or fully-unstacked broadcast patterns (rejects partial broadcast)
    - Elementwise/activations: rank <= 4; elementwise requires identical shapes
- `onnxruntime/core/providers/telum/graph_transformers/telum_transformer_base.h`
  - Avoids null deref by skipping optional inputs.

Assessment:

- Capability gating is critical for correctness, especially when CPU fallback is disabled in tests.
- Current gating still has a mismatch with actual kernel availability (see "Issues" section below).

### Test integration (Telum-only test binary)

State: **Wired** when Telum is enabled; tests now actually execute Telum kernels.

Relevant files:

- Build wiring:
  - `cmake/onnxruntime_unittests.cmake`
    - Adds `add_subdirectory(test/providers/telum)` when `onnxruntime_USE_TELUM`.
  - `onnxruntime/test/providers/telum/CMakeLists.txt`
    - Builds `onnxruntime_telum_test` and links zDNN and `onnxruntime_providers_telum`.
- Test behavior fixes:
  - `onnxruntime/test/providers/telum/test_utils.h`
    - Adds `RunOnTelum(...)` helper:
      - explicitly appends Telum EP
      - sets `session.disable_cpu_ep_fallback=1` by default
      - runs via `OpTester::RunWithConfig()`
- `onnxruntime/test/util/include/default_providers.h`
  - Adds `DefaultTelumExecutionProvider()` under `USE_TELUM`.

Assessment:

- This is the right testing strategy: force Telum execution or fail, and skip when NNPA isn't present.

### Build.py integration (CLI flags)

State: **Integrated**.

Relevant files:

- `tools/ci_build/build_args.py`
  - Adds `--use_telum` and `--telum_home` (falls back to env `ZDNN_ROOT`).
- `tools/ci_build/build.py`
  - Threads options into CMake defines:
    - `-Donnxruntime_USE_TELUM=ON`
    - `-DZDNN_ROOT=...`

Assessment:

- Correct direction.
- Ensure docs refer to `onnxruntime_USE_TELUM` (lowercase prefix), not `ONNXRUNTIME_USE_TELUM`.

## Are We On The Right Track?

Yes, with a few high-impact corrections and follow-on phases needed.

Reasons this is on the right track:

- **In-tree EP + standard selection**: It uses the normal ORT mechanisms (provider factory creators, generic append, python provider list).
- **Tight build gating**: Telum stays off by default; turning it on forces s390x and zDNN.
- **Kernel registry is real**: `GetCapability()` can actually claim nodes.
- **zDNN semantics are being treated seriously**:
  - MatMul bias requirement is enforced.
  - Unsupported broadcast patterns are rejected up-front.
- **Tests are converging to "real execution"**:
  - disabling CPU EP fallback is the key move that prevents false positives.

## Things That Should Be Done Differently (Or Tightened Up)

### 1) Align supported-op declarations with reality

Problem:

Historically, `TelumExecutionProvider` declared support for `Softmax` and `LayerNormalization` before kernels existed.

Impact:

If supported-op lists drift from kernel reality, `GetCapability()` behavior becomes hard to reason about.

Recommendation:

Keep `supported_ops_`, kernel registry, and per-op capability gating in sync at all times.

### 2) Data type support is inconsistent (BFLOAT16)

Problem:

Historically, `ValidateDataTypes()` allowed BFLOAT16 while some kernels only registered `float` and `MLFloat16`.

Impact:

Partitioning can silently skip nodes due to kernel lookup failures that are hard to diagnose.

Recommendation:

If Telum supports BF16, wire it end-to-end:

- kernel type constraints
- any CPU post-processing paths
- tests that force Telum execution with BF16

Transformer implication:

- BF16 support is likely important on IBM Z.

### 3) "Correct-but-slow" transform-per-inference is OK for bring-up, but not for production

Problem:

- Kernels transform ORT tensors to zDNN format every `Compute()` call.
- For constants (weights/bias), this repeats expensive transformations each inference.

Impact:

- Major performance loss, especially for transformer linear layers.

Recommendation:

- Implement `PrePack(...)` on MatMul/Gemm kernels to:
  - detect constant/initializer weights and bias
  - allocate and store transformed ztensors in kernel state
  - reuse them on each invocation

### 4) Elementwise broadcasting is a known zDNN gap and must be addressed for full transformer coverage

Problem:

- zDNN elementwise ops do not broadcast, and the current Telum elementwise kernels reject any broadcast case.

Impact:

- Many ONNX graphs rely on broadcasting (bias vectors, scalar adds/muls, etc.).
- ORT graph transformers (e.g., MatMul+Add -> Gemm) will cover some but not all cases.

Recommendation:

- For phase 1, implement only the broadcast patterns needed for transformer bring-up:
  - bias vector `[H]` or `[1,H]` across `[..., H]`
  - scalar
  - optionally `[B,1,H]` and friends (depending on exported models)
- Do the broadcast in-kernel (CPU) when needed, but still keep the node on Telum EP so partitioning stays stable.

### 5) Telum-specific graph transformer code is currently unused

Problem:

- `onnxruntime/core/providers/telum/graph_transformers/linear_fusion.cc` exists, but `RegisterGraphTransformers()` is currently a stub.
- ORT core already has `MatMulAddFusion` (`onnxruntime/core/optimizer/matmul_add_fusion.cc`) that handles many MatMul+Add -> Gemm cases.

Recommendation:

- Either:
  - delete the Telum-specific transformer and rely on core fusions, or
  - wire it up only if it provides unique value not covered by ORT core fusions.

### 6) Documentation must be corrected to match the actual APIs and build knobs

Problem:

- `onnxruntime/core/providers/telum/README.md` currently documents:
  - wrong CMake flag name (`ONNXRUNTIME_USE_TELUM` vs `onnxruntime_USE_TELUM`)
  - non-existent C++ wrapper APIs (`AppendExecutionProvider_Telum()`)
  - ops that are not implemented (Softmax/LayerNormalization)

Recommendation:

- Update Telum README to be accurate and avoid claiming support that doesn't exist.
- Make the docs explicit about:
  - supported ops and constraints
  - how to run tests
  - how to validate you actually executed on Telum (disable CPU fallback)

## Risk Register (What Could Bite Us Later)

These are not necessarily blockers for phase 1, but should be tracked.

1. **Numerics**:
   - Softmax and layernorm are numerically sensitive; FP16/BF16 paths need careful tolerance and reference computations.
2. **Shape coercion correctness**:
   - Logical shape reinterpretation is correct if and only if flattening order matches. Current MatMul approach is correct for row-major flattening of batch dims; apply same rigor to softmax/layernorm.
3. **Session-level caching / memory**:
   - Prepacked ztensors increase memory usage; need eviction or sharing strategy if many weights exist.
4. **CI coverage**:
   - Without an s390x CI runner, regressions will slip in.
5. **Operator coverage**:
   - Real transformer graphs may require additional ops beyond the P0/P1 list (e.g., `ReduceMean`, `Reshape`, `Transpose`, `Attention` fusions).

## Recommended Phased Plan (High Level)

See `docs/telum/Telum_EP_TODO.md` for a concrete checklist. At a high level:

1. Phase 0: Hygiene and consistency
   - Fix docs, supported-op declarations, BF16 policy, and validate non-Telum builds.
2. Phase 1: Transformer-critical ops + tests
   - Implement Softmax and LayerNormalization kernels (or the subset Telum hardware can support).
   - Add tests that force Telum execution.
3. Phase 2: Broadcast + shape-interop edges for real models
   - Add limited broadcast patterns that real transformer exports require.
4. Phase 3: Performance
   - Prepack weights/bias; avoid per-run transforms for constant tensors.
5. Phase 4: Packaging/CI
   - Add an s390x build/test job and/or documented reproducible environment to run Telum tests.
