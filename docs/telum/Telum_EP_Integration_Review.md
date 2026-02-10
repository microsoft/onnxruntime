# Telum Execution Provider (s390x/z16) Integration Review

Date: 2026-02-10

Repo: `onnxruntime/` (local workspace)

Branch reviewed (local): `feature/telum-ep-integration-next` (working tree at review time)

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
- **ztensor descriptor lifetime**: `zdnn_init_ztensor` stores pointers to `zdnn_tensor_desc` structs; the descriptor storage must outlive the `zdnn_ztensor`. Telum EP must not use stack-allocated descriptors.

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
    - Elementwise: rank <= 4; ONNX/Numpy broadcasting supported in-kernel on CPU (zDNN path used when shapes match)
    - Activations: rank <= 4
    - `Softmax`: axis == last dim only
    - `LayerNormalization`: axis == last dim only; scale/bias shape [C]

Assessment:

- Capability gating is critical for correctness, especially when CPU fallback is disabled in tests.
- Capability gating should continue to be kept in sync with the Telum kernel registry as operator coverage expands.

### Strict Mode (Targeted "No Silent CPU" Guardrail)

State: **Implemented**.

Relevant files:

- `onnxruntime/core/providers/telum/telum_execution_provider.cc`
  - When `strict_mode=true`, Telum throws during `GetCapability()` if it encounters a node whose `OpType()` is in
    Telum's `supported_ops_` list but cannot be supported due to:
    - dynamic shapes
    - unsupported input types
    - unsupported per-op shape/broadcast/attribute constraints
    - missing Telum kernel registration for the node's domain/version/types

Assessment:

- This is useful to catch "I thought I was running on Telum but I'm not" cases early at session init time.
- This is complementary to `session.disable_cpu_ep_fallback=1`:
  - `disable_cpu_ep_fallback` is a global "no CPU nodes" guardrail.
  - `strict_mode` is a Telum-scoped guardrail for op types we intend to accelerate.

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
- In addition to per-op `OpTester` coverage, there is an end-to-end multi-op graph test that verifies:
  - Telum can take the entire graph (no CPU node assignment)
  - Outputs match CPU EP within a tight tolerance
  - file: `onnxruntime/test/providers/telum/test_end_to_end.cc`

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

### CI Coverage (s390x Self-Hosted)

State: **Added** (requires a self-hosted GitHub Actions runner on `linux/s390x` with zDNN installed).

Relevant files:

- `.github/workflows/linux_s390x_telum_ci.yml`
  - Builds ORT with `--use_telum` and `--telum_home="$ZDNN_ROOT"`.
  - Runs `ctest -L telum -V`.

Assessment:

- This is the correct direction: Telum coverage needs real s390x hardware in CI to avoid regressions.

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

### 2) Data type support must remain consistent across gating, kernels, and tests (BFLOAT16)

Problem:

If Telum's `ValidateDataTypes()` accepts a type that no matching Telum kernel is registered for, Telum may skip nodes
due to kernel lookup failures that are hard to diagnose.

Current status:

- BFLOAT16 is wired end-to-end across Telum kernels (type constraints, CPU post-processing where needed) and tests.

Recommendation:

Treat type support as an "all layers must agree" contract:

- capability gating (`ValidateDataTypes()` and per-op checks)
- kernel registry type constraints
- any CPU post-processing paths inside kernels
- tests that force Telum execution for each supported type

### 3) "Correct-but-slow" transform-per-inference is OK for bring-up, but not for production

Problem:

- Kernels transform ORT tensors to zDNN format every `Compute()` call.
- For constants (weights/bias), this repeats expensive transformations each inference.

Impact:

- Major performance loss, especially for transformer linear layers.

Current status:

- `PrePack(...)` is implemented for:
  - `MatMul` constant RHS (B) initializers
  - `Gemm` constant weight matrix (B) initializers
  - `Gemm` bias vector (C) initializers for the safe fused subset (`alpha==beta==1` and bias vector shape)

Remaining (performance polish):

- Consider caching/reusing the "zero bias" ztensors that are currently created per-inference when no bias is fused.
  - This is a pure performance optimization; correctness is already covered by explicit zero bias creation.

### 4) Elementwise broadcasting is a known zDNN gap and must be addressed for full transformer coverage

Problem:

- zDNN elementwise ops do not broadcast.
- ONNX graphs frequently rely on broadcasting (bias vectors, scalars, etc.).

Impact:

- Many ONNX graphs rely on broadcasting (bias vectors, scalar adds/muls, etc.).
- ORT graph transformers (e.g., MatMul+Add -> Gemm) will cover some but not all cases.

Recommendation:

- For phase 1, implement only the broadcast patterns needed for transformer bring-up:
  - bias vector `[H]` or `[1,H]` across `[..., H]`
  - scalar
  - optionally `[B,1,H]` and friends (depending on exported models)
- Do the broadcast in-kernel (CPU) when needed, but still keep the node on Telum EP so partitioning stays stable.

Current status:

- Telum elementwise kernels now implement ONNX/Numpy broadcasting on a CPU path for rank <= 4 (and use zDNN when shapes match).
- This is intentionally "correct first"; we can decide later whether to expand rank > 4 broadcast support.

### 5) Telum-specific graph transformer code should not be carried unless it is wired (resolved)

Problem:

- Telum-specific fusion scaffolding was added, but it was not actually wired into the ORT optimizer pipeline.
- ORT core already has `MatMulAddFusion` (`onnxruntime/core/optimizer/matmul_add_fusion.cc`) that handles many MatMul+Add -> Gemm cases.

Current status:

- The Telum-specific graph transformer scaffolding was removed to avoid carrying dead code in-tree.
- Telum relies on ORT core fusions (like `MatMulAddFusion`) to produce `Gemm` where applicable.

Recommendation:

- If Telum needs fusions beyond ORT core, integrate them via supported ORT hooks (graph optimizer registry and/or
  compile-based fusion), not via unregistered transformer files.

### 6) Documentation must match the actual knobs, APIs, and constraints (mostly resolved)

Problem:

Docs that claim support that doesn't exist or reference the wrong build flags lead to wasted time and incorrect usage.

Current status:

- `onnxruntime/core/providers/telum/README.md` has been updated to reflect:
  - the correct CMake flag (`onnxruntime_USE_TELUM`)
  - the `ZDNN_ROOT` requirement
  - the actual selection APIs (Python provider list and generic C/C++ append-by-name)
  - the real supported op set and constraints
  - how to run Telum tests and prove you executed Telum kernels (disable CPU fallback)

Remaining:

- Add a top-level execution provider doc entry (under the standard `docs/execution_providers/` tree) that links to the
  Telum EP README and documents the build and runtime knobs at the same level as other EPs.

### 7) Telum EP should not override `Compile()` unless it actually compiles fused subgraphs (fixed)

Problem:

Kernel-based EPs generally do not need to override `IExecutionProvider::Compile(...)`. A stub `Compile()` that returns
success but does not generate correct compute functions is dangerous if it is ever invoked.

Current status:

- Telum no longer overrides `Compile()`. If ORT ever attempts to call `Compile()` for Telum, the base implementation
  returns `NOT_IMPLEMENTED` instead of silently producing incorrect results.

### 8) zDNN ztensor descriptor lifetime must be handled correctly (fixed)

Problem:

`zdnn_init_ztensor` stores pointers to `zdnn_tensor_desc` fields (it does not deep-copy the descriptors). If a Telum kernel initializes ztensors with stack-allocated descriptors, those pointers dangle after the function returns and zDNN calls can crash or corrupt memory.

Current status:

- Telum `TensorConverter` now heap-allocates the descriptors and transfers ownership to the ztensor.
- `TelumKernel::ZTensorGuard` frees both the ztensor buffer and the heap-allocated descriptors.

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
