# WebGPU MatMul Algorithm Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace conditional trial dispatch in `ComputeMatMul` with an explicit, vendor-extensible algorithm scheduler and a test-only forced-algorithm session option.

**Architecture:** A lightweight algorithm enum is shared by WebGPU provider configuration and MatMul. A pure scheduler preserves current precedence and delegates vendor rules to an Intel subclass; `ComputeMatMul` dispatches the chosen enum through one switch and validates hard prerequisites when a test forces a path.

**Tech Stack:** C++17, ONNX Runtime WebGPU EP, GoogleTest, CMake/Visual Studio on Windows

**Spec:** `docs/superpowers/specs/2026-09-16-webgpu-matmul-algorithm-scheduler-design.md`

## Global Constraints

- Automatic selection must preserve current behavior and rule precedence.
- Forced selection bypasses heuristics but never bypasses correctness or hardware prerequisites.
- Unsupported forced algorithms fail explicitly and never silently fall back.
- `PackedSplitK` is independently selectable from `Packed`.
- The option remains internal/test-only and is not added to public documentation.
- Existing MatMul, pointwise Conv, and Attention callers continue through `ComputeMatMul`.
- The Windows validation build enables Dawn Vulkan, disables Dawn D3D12, and tests explicitly select `dawnBackendType=Vulkan`.

---

### Task 1: Algorithm Identity and Configuration Plumbing

**Files:**
- Create: `onnxruntime/core/providers/webgpu/math/matmul_algorithm.h`
- Modify: `onnxruntime/core/providers/webgpu/webgpu_provider_options.h`
- Modify: `onnxruntime/core/providers/webgpu/webgpu_execution_provider.h`
- Modify: `onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc`
- Modify: `onnxruntime/core/providers/webgpu/webgpu_provider_factory.cc`
- Modify: `onnxruntime/core/providers/webgpu/compute_context.h`
- Test: `onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc`

**Interfaces:**
- Produces: `enum class MatMulAlgorithm { SubgroupMatrix, Naive, IntelSubgroup, Packed, PackedSplitK }`.
- Produces: `std::optional<MatMulAlgorithm> ParseMatMulAlgorithm(std::string_view)` and `std::string_view MatMulAlgorithmName(MatMulAlgorithm)`.
- Produces: `ComputeContextBase::ForcedMatMulAlgorithm() const` returning `std::optional<MatMulAlgorithm>`.

- [ ] **Step 1: Write the failing enum parsing tests**

```cpp
EXPECT_EQ(ParseMatMulAlgorithm("packed_split_k"), MatMulAlgorithm::PackedSplitK);
EXPECT_EQ(MatMulAlgorithmName(MatMulAlgorithm::PackedSplitK), "packed_split_k");
EXPECT_EQ(ParseMatMulAlgorithm("unknown"), std::nullopt);
```

- [ ] **Step 2: Build to verify the new tests fail**

```powershell
python tools/ci_build/build.py --config RelWithDebInfo --build_dir .\build\WGPU-Vulkan --use_webgpu --update --build --parallel --target onnxruntime_provider_test --cmake_extra_defines onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF
```

Expected: compilation fails because the algorithm interface is missing.

- [ ] **Step 3: Implement enum conversion and option plumbing**

Use exact case-sensitive names `subgroup_matrix`, `naive`, `intel_subgroup`, `packed`, and `packed_split_k`. Declare `kForceMatMulAlgorithm = "ep.webgpuexecutionprovider.forceMatmulAlgorithm"`, parse it in `ParseEpConfig`, reject invalid values with the accepted-value list, store an `std::optional<MatMulAlgorithm>` in `WebGpuExecutionProvider`, and expose it through `ComputeContextBase`.

- [ ] **Step 4: Build and run parser tests**

Build with Step 2, locate `onnxruntime_provider_test.exe`, and run `--gtest_filter="MatMulAlgorithmParsingTest.*"` from its directory. Confirm a non-zero test count and zero failures.

- [ ] **Step 5: Commit**

```powershell
git add onnxruntime/core/providers/webgpu onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc
git commit -m "webgpu: add forced MatMul algorithm option"
```

### Task 2: Common and Intel Algorithm Schedulers

**Files:**
- Modify: `onnxruntime/core/providers/webgpu/math/matmul.h`
- Modify: `onnxruntime/core/providers/webgpu/math/matmul.cc`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.h`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.cc`
- Test: `onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc`

**Interfaces:**
- Consumes: `MatMulAlgorithm` and the optional forced value from Task 1.
- Produces: `MatMulAlgorithmSelectionParams` with common rule facts.
- Produces: `MatMulAlgorithmScheduler::Select(const MatMulAlgorithmSelectionParams&, std::optional<MatMulAlgorithm>) const`.
- Produces: an Intel scheduler overriding the vendor rule hook.

- [ ] **Step 1: Write failing scheduler tests**

Cover forced precedence, subgroup-matrix precedence, exact `N < 8 && K < 8` boundaries, Intel selection, Split-K selection, and packed fallback.

```cpp
EXPECT_EQ(base.Select(small, std::nullopt), MatMulAlgorithm::Naive);
EXPECT_EQ(intel.Select(intel_problem, std::nullopt), MatMulAlgorithm::IntelSubgroup);
EXPECT_EQ(base.Select(split_k_problem, std::nullopt), MatMulAlgorithm::PackedSplitK);
EXPECT_EQ(base.Select({}, MatMulAlgorithm::Packed), MatMulAlgorithm::Packed);
```

- [ ] **Step 2: Build and confirm compilation fails for missing scheduler types**

Run the Task 1 build command and retain the compiler failure as the red TDD result.

- [ ] **Step 3: Implement the common scheduler**

Implement a non-virtual `Select` that checks forced selection, subgroup matrix, naive, a protected virtual vendor hook, Split-K, and packed in that order. The base vendor hook returns `std::nullopt`.

- [ ] **Step 4: Implement the Intel scheduler**

The Intel override returns `IntelSubgroup` only for the supplied current-rule fact. Add a scheduler factory selected by `context.AdapterInfo().vendor`.

- [ ] **Step 5: Build, run `--gtest_filter="MatMulAlgorithmSchedulerTest.*"`, and commit**

```powershell
git add onnxruntime/core/providers/webgpu/math/matmul.h onnxruntime/core/providers/webgpu/math/matmul.cc onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.h onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.cc onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc
git commit -m "refactor: add WebGPU MatMul algorithm scheduler"
```

### Task 3: Direct Enum Dispatch

**Files:**
- Modify: `onnxruntime/core/providers/webgpu/math/matmul.h`
- Modify: `onnxruntime/core/providers/webgpu/math/matmul.cc`
- Modify: `onnxruntime/core/providers/webgpu/math/subgroup_matrix_matmul.cc`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/gemm_subgroup.h`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/gemm_subgroup.cc`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.h`
- Modify: `onnxruntime/core/providers/webgpu/vendor/intel/math/matmul.cc`
- Test: `onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc`

**Interfaces:**
- Produces: `MatMulOptImpl::CanApply(...) const` and `Compute(...)` without a `handled` output.
- Produces: separate Intel hard-capability and automatic-heuristic checks.
- Produces: one `switch (algorithm)` in `ComputeMatMul` and one helper per implementation.

- [ ] **Step 1: Write failing prerequisite tests**

Cover forced Split-K rejection for deterministic compute and incompatible packing/activation/layout. Cover Intel hard capability separately from its `M >= 64 && N >= 512 && K >= 32` automatic heuristic.

- [ ] **Step 2: Run focused tests and preserve the red result**

Run `onnxruntime_provider_test.exe --gtest_filter="MatMulAlgorithm*"`; expect missing validation interfaces or failing assertions.

- [ ] **Step 3: Split subgroup-matrix applicability from execution**

Move non-mutating early-decline checks into `CanApply`, make `Compute` error if called inapplicably, remove `handled`, and preserve odd-N cached padding.

- [ ] **Step 4: Separate Intel capability from heuristic policy**

Hard capability is `vendor == intel && Subgroups`. Automatic policy adds the existing `M >= 64 && N >= 512 && K >= 32` thresholds. Forced mode uses only hard capability.

- [ ] **Step 5: Extract naive and packed helpers**

Make the packed helper accept explicit `bool use_split_k`. False never runs the Split-K heuristic; true validates a configured split size, non-deterministic compute, vec4 packing, no fused activation, and compatible bias layout before atomic accumulation.

- [ ] **Step 6: Replace conditional execution with one switch**

```cpp
switch (algorithm) {
  case MatMulAlgorithm::SubgroupMatrix: return subgroup_impl->Compute(...);
  case MatMulAlgorithm::Naive: return ApplyMatMulNaive(...);
  case MatMulAlgorithm::IntelSubgroup: return intel::ApplyMatMulIntel(...);
  case MatMulAlgorithm::Packed: return ApplyMatMulPacked(..., false);
  case MatMulAlgorithm::PackedSplitK: return ApplyMatMulPacked(..., true);
}
```

Every forced-prerequisite error includes `MatMulAlgorithmName(algorithm)`.

- [ ] **Step 7: Build, run `--gtest_filter="MatMulAlgorithm*"`, and commit**

```powershell
git add onnxruntime/core/providers/webgpu/math onnxruntime/core/providers/webgpu/vendor/intel/math onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc
git commit -m "refactor: dispatch WebGPU MatMul by algorithm"
```

### Task 4: Forced-Path Integration and Verification

**Files:**
- Modify: `onnxruntime/test/providers/webgpu/matmul_large_test.cc`
- Modify: `onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc`

**Interfaces:**
- Consumes: `kForceMatMulAlgorithm` and `WebGpuExecutionProviderWithOptions`.
- Produces: end-to-end proof that forcing changes dispatch and preserves output.

- [ ] **Step 1: Write failing integration tests**

Create each EP with `kDawnBackendType=Vulkan`. Use a non-small shape forced to `Naive`, a small shape forced to `Packed`, a compatible large shape forced to `PackedSplitK`, and a compatible f16 shape forced to `SubgroupMatrix`. Add an expected-failure case with deliberately unmet hard prerequisites.

- [ ] **Step 2: Run `--gtest_filter="WebGpuMatMulAlgorithmTest.*"` and preserve the failing result**

Confirm a non-zero selected test count and at least one expected pre-implementation failure.

- [ ] **Step 3: Complete only wiring or diagnostics exposed by the tests**

Ensure test EP construction passes the option through `ConfigOptions`, the same path used by WebGPU session creation.

- [ ] **Step 4: Run focused and regression tests**

```powershell
.\onnxruntime_provider_test.exe --gtest_filter="MatMulAlgorithm*:WebGpuMatMulAlgorithmTest.*"
.\onnxruntime_provider_test.exe --gtest_filter="MathOpTest.MatMulFloatType:MathOpTest.MatMul_Float16:WebGpuMatMulLargeTest.*"
```

Confirm non-zero counts and zero failures. The subgroup-matrix case must verify that the Vulkan adapter advertises the required subgroup-matrix configuration and subgroup-size control; otherwise it may skip only with the exact missing capability in its reason.

- [ ] **Step 5: Run formatting and diff checks**

Run the repository C++ formatter/linter for changed files, then:

```powershell
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
```

- [ ] **Step 6: Commit and report limitations**

```powershell
git add onnxruntime/test/providers/webgpu/matmul_large_test.cc onnxruntime/test/providers/webgpu/matmul_algorithm_scheduler_test.cc
git commit -m "test: force WebGPU MatMul algorithms"
```

Report exact commands, counts, Vulkan adapter identity, and observed subgroup-matrix capabilities. State that macOS-arm64 Metal CI remains additional cross-backend validation and that lavapipe cannot execute this family reliably.
