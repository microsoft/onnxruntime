---
name: webgpu-ep-review
description: Review pull requests, branches, and patches that change the native WebGPU execution provider, including kernels and WGSL generation under onnxruntime/core/providers/webgpu and onnxruntime/contrib_ops/webgpu, .wgsl.template files, program cache keys, dispatch and workgroup sizing, buffer management, graph capture, WebGpuContext, the WebGPU plugin EP, WebGPU kernel registrations, and WebGPU tests. Use together with the code-review skill.
---

# Reviewing Native WebGPU EP Changes

Use this skill with `/code-review`, which owns the general workflow and the report format. The invariants are in
`.github/instructions/webgpu-ep.instructions.md`; this skill is the procedure for checking them against a diff. It
does not cover JSEP (`onnxruntime/core/providers/js/`, `js/web/lib/wasm/jsep/`), which has its own instructions.

## 1. Classify the Change

| Change | Typical files | Sections |
|---|---|---|
| Kernel or shader | A `Program` or `WebGpuKernel` subclass, `GenerateShaderCode`, `*.wgsl.template` | 2–5, 7, 8 |
| Registration or type constraints | `webgpu_execution_provider.cc`, `webgpu_contrib_kernels.cc`, kernel macros | 7, 8 |
| EP infrastructure | `webgpu_context.*`, `program_manager.*`, `buffer_manager.*`, `allocator.*`, `data_transfer.*`, `session_buffer_pool.*`, `compute_context.*`, `ep/`, `webgpu_provider_factory.cc` | 6, 8 |
| Build, Dawn, packaging | `cmake/onnxruntime_providers_webgpu.cmake`, `cmake/patches/dawn/`, `plugin-ep-webgpu/` | 6 |
| Tests only | `onnxruntime/test/providers/webgpu/`, shared `OpTester` tests | 8 |

## 2. Trace Each Changed Kernel

Before judging a changed `Program`, trace it end to end. Read the base version of every function the diff touches,
not only the hunk: cache-key and bounds defects usually involve lines the diff did not change.

1. **Host side (`ComputeInternal`).** Which inputs are read on the host, what is validated, how the output shape is
   computed, which early exits exist, and how 64-bit values are narrowed for uniforms.
2. **Program setup.** Each `AddInput` and `AddOutput` with its `ProgramTensorMetadataDependency` and component count,
   `SetDispatchGroupSize`, `SetWorkgroupSize`, `SetSubgroupSize`, `CacheHint`, `AddUniformVariables`, overridable
   constants, and `SetIndirectDispatchTensor`.
3. **Shader side.** `GenerateShaderCode`, or the `.wgsl.template` plus the parameters C++ passes to it. List every C++
   value that changes the emitted text.
4. **Registration.** Macro, domain, opset range, type constraints, `InputMemoryType(OrtMemTypeCPUInput, ...)`
   declarations, and the entry in the registration table.
5. **Tests.** Which test exercises each shader variant on WebGPU.

## 3. Cache-Key Audit

This check has the highest value at review time. The nightly job (`nightly_webgpu.yml`) runs the tests with a shader
dump and validates keys only after merge.

1. List every value the shader text depends on: members and attributes, data types, rank and component counts,
   vectorization decisions derived from shapes, feature or vendor branches, and provider options.
2. For each value, find what keys it: `CacheHint`, a declared dependency (`Type` keys element types, `Rank` keys the
   rank, `Shape` keys the dimensions), the component count, the workgroup size, or overridable constants. Confirm
   against `CalculateProgramCacheKey` in `program_cache_key.cc` rather than assuming.
3. Report any unkeyed value as blocking. Give the trigger: two calls that differ only in that value, share a key,
   and therefore share a pipeline.

The program cache belongs to a `WebGpuContext`. Values that are fixed for a device, such as adapter features and
limits, need no key. Values that can differ between sessions sharing a context, such as provider options, do.

Common misses:

- Concrete dimensions in WGSL (loop bounds, unrolled loops, `const` arrays) while the dependency is `TypeAndRank` or
  `None`.
- A branch on component count or a shape-derived fast path without a matching key component.
- A new member used in `GenerateShaderCode` while `CacheHint` is unchanged.
- Floats keyed with `std::to_string`, which is lossy. `onnxruntime/test/providers/webgpu/activation_cache_key_test.cc`
  shows the expected style of key test.
- A template parameter added to a `.wgsl.template` call without a key component.

The reverse, a key component that does not change the WGSL, only duplicates pipelines. Report it only when it causes
measurable compile churn.

## 4. Dispatch, Bounds, and WGSL Hazards

- **Addressing.** Linear kernels must use `global_idx`, `workgroup_idx`, or `local_idx`. A kernel that addresses by
  `workgroup_id.x`, `.y`, or `.z` must keep each dispatch dimension at or below `maxComputeWorkgroupsPerDimension`;
  otherwise `NormalizeDispatchGroupSize` reshapes the grid and the mapping is wrong.
- **Guards.** Every invocation past the logical size must be inactive before any memory access. A kernel that uses
  workgroup memory or barriers must predicate the work instead of returning early.
- **Partial tiles.** Check M, N, and K remainders on loads and stores, including subgroup-matrix tiles.
- **Degenerate shapes.** Check zero-size outputs, which must return before dispatch, rank-0 scalars, dimensions of 1
  under broadcasting, and empty spatial dimensions.
- **Uniform arrays.** Index uniform arrays with the `GetElementAt` helper (`getElementAt` in `.wgsl.template` files)
  so the vec4 packing of uniform arrays is honored.
- **Resource limits.** Compute workgroup memory from the tile constants for both f32 and f16 variants and compare it
  with 16 KiB, or with the limit the code queries. Count storage-buffer bindings against
  `maxStorageBuffersPerShaderStage`.
- **Uniformity.** Look for early returns, invocation-dependent loop trip counts, and divergent branches around
  `workgroupBarrier`, `storageBarrier`, and subgroup operations. Code that assumes a subgroup size must request it
  with `SetSubgroupSize`, which requires the subgroup-size-control feature, or handle every size.
- **`select`.** WGSL evaluates both operands of `select`, so an out-of-range index in the unselected operand still
  reads memory. Clamp the index.
- **Integer semantics.** WGSL signed `/` and `%` truncate toward zero. ONNX operations with floor semantics, such as
  `Mod` with `fmod=0`, and negative-index wrapping need explicit code.
- **Precision.** f16 sums overflow above 65504. Reductions and dot products should accumulate in f32 unless the
  operation's precision contract says otherwise. `enableMatmulFp32Accumulation` is the switch for MatMul, Gemm, and
  MatMulNBits.
- **NaN and rounding.** WGSL `min`, `max`, and `clamp` do not guarantee NaN propagation. WGSL `round` rounds half to
  even, so other tie rules need explicit code.

## 5. Host Validation and Narrowing

- Search the changed host code for `static_cast<int32_t>`, `static_cast<uint32_t>`, `gsl::narrow_cast`, and C-style
  casts applied to shapes, attributes, or sizes that reach `AddUniformVariables`. Each needs a range check or
  `onnxruntime::narrow`.
- Element and byte counts must use overflow-safe arithmetic. A tensor with more than 2^32 elements cannot be described
  by a `u32` uniform; it must fail with a clear `Status` or use segmented access.
- Every attribute or host input that feeds sizes or indices needs validation of range, sign, rank, and divisibility,
  returned as a `Status` rather than thrown.
- Each host-read input must be declared `OrtMemTypeCPUInput` in every registration of the operator, across all opset
  ranges.
- For every index computed from tensor data on the GPU, find the clamp or skip in the WGSL. Compare with the
  RotaryEmbedding `position_ids` bounds validation.

## 6. Infrastructure, Concurrency, Graph Capture, Portability

- **Dawn callbacks.** Callbacks should be `noexcept` and must not use `ORT_ENFORCE`, `ORT_THROW`, or code that
  throws. The callback context must outlive the future; see the comments in `program_manager.h`.
- **Concurrency.** `ComputeInternal` is `const`. Flag `mutable` members, function-local statics that hold device
  objects or per-run data, and unsynchronized global caches. Sessions can run concurrently on one context.
- **Graph capture.** Clears, copies, and uploads issued outside `RunProgram` must go through context APIs that graph
  capture records. Temporaries must come from the context's allocation path, not raw `wgpu::Device` calls, so their
  lifetime matches a captured graph. Changes to the buffer manager, allocator, session buffer pool, or data transfer
  affect every kernel and need targeted tests.
- **Features and vendors.** Every use of an optional feature needs a device-support check and a fallback that tests
  cover. Configuration tables filled only in native builds break WebAssembly builds.
- **Plugin EP.** New dependencies from WebGPU EP code on ONNX Runtime internals outside the EP must also be available
  to the plugin EP build. The Windows WebGPU workflow compiles the plugin EP.
- **Dawn updates.** Changes under `cmake/patches/dawn/` need evidence from every backend the change can affect.

## 7. Registration and Documentation

- Opset ranges must be contiguous and must not overlap. When a new version is added, the previous registration
  becomes a versioned range that ends at the new version minus one, and both the forward declaration and the
  `BuildKernelCreateInfo` entry change with it.
- Layout-sensitive operators can also be registered in `kMSInternalNHWCDomain` for the NHWC preferred layout. A change
  to one registration usually requires the other.
- Type constraints must match the shader. WGSL has no 64-bit integer type; a new int64 path must state and test its
  range assumption and follow the existing `GetOpTypeConstraints` and `enableInt64` pattern.
- Registration changes require a regenerated `docs/OperatorKernels.md`. The `windows_gpu_doc_gen.yml` workflow
  validates it with WebGPU enabled.

## 8. Tests and Validation Evidence

Require a test that runs on WebGPU for each behavior change and each new shader variant: f16 and f32, vectorized and
scalar paths, every supported rank, broadcast and non-broadcast cases, empty tensors, and large tensors when indexing
changes. Shared `OpTester` tests run on WebGPU when it is built, unless the test excludes it.
`onnxruntime/test/providers/webgpu/` holds WebGPU-specific tests such as `conv_large_index_test.cc`,
`matmul_large_test.cc`, and `activation_cache_key_test.cc`.

- An added exclusion of `kWebGpuExecutionProvider`, or a looser tolerance, needs a stated reason. An f16 tolerance can
  be justified by precision; an f32 tolerance usually hides a defect.
- CI leaves gaps: the Linux WebGPU job only builds, the Windows WebGPU tests run on NVIDIA, and macOS runs on Apple
  GPUs. Changes to Intel, AMD, or Qualcomm paths need results from that hardware in the PR description.
- Performance claims need before and after numbers with the model, device, and method (`docs/PR_Guidelines.md`).

Validation to request or run:

```bash
./build.sh --config Release --parallel --use_webgpu          # see the ort-build skill
cd build/Linux/Release
./onnxruntime_provider_test --gtest_filter="*<OpName>*"         # see ort-test; no GPU: webgpu-local-testing
```

For a change to shader generation or cache keys, run the same validation as the nightly job:

```bash
export ORT_WEBGPU_EP_SHADER_DUMP_FILE=/tmp/webgpu_shaders.log   # PowerShell: $env:ORT_WEBGPU_EP_SHADER_DUMP_FILE=...
./onnxruntime_provider_test --gtest_filter="*<OpName>*"
node .github/actions/webgpu-validate-shader-key/validate-shader-key.js < /tmp/webgpu_shaders.log
```

The dump file also holds the generated WGSL for each program and key. To surface WebGPU validation errors while
debugging, set the WebGPU EP option `validationMode` (config key `ep.webgpuexecutionprovider.validationMode`) to
`full`. Release builds default to `disabled`. For template generator changes, run
`python tools/python/wgsl_template/test/run_tests.py`.

## 9. Regression Patterns From Merged Fixes

| Pattern | Fixed in | Check |
|---|---|---|
| int64 attribute truncated to int32, then an out-of-bounds read | #28721 (Pad) | Section 5 narrowing |
| Out-of-bounds subgroup-matrix loads and stores on trailing partial tiles | #32364, #32582 | Partial tiles |
| Dispatching an empty output | #33668 (pooling) | Degenerate shapes |
| Rank-0 and empty-rank inputs | #33664 (Gemm scalar bias), #33171 (MatMul, DequantizeLinear) | Degenerate shapes |
| Dispatch-shape specialization in the shader | #32343 (1D-dispatch fast path) | Addressing, cache key |
| Exceptions thrown from Dawn callbacks | #29625 | Dawn callbacks |
| Missing host validation of degenerate inputs | #29255 (GridSample), #29595 (LSTM) | Section 5 |
| GPU-read indices used without bounds | RotaryEmbedding `position_ids` validation | Section 5 |
| Rounding tie rules | #28757 (Resize nearest) | NaN and rounding |
| Tensors larger than one storage binding | #29463 (Conv offset helpers) | Resource limits |
| Shader-key collisions | #33153 (collision diagnostics), #33693 (nightly check) | Section 3 |
| State shared across concurrent sessions | #32587 | Concurrency |
| GPU work outside programs missing from graph capture | #33509 (clears), #33362 (copies) | Graph capture |
| Configuration filled only in native builds | Subgroup-matrix configuration fix for WebAssembly | Features and vendors |

## 10. Reporting

Report findings in the `/code-review` format: file and line, evidence or a triggering scenario, impact, and a
correction. For WebGPU findings, also:

- Name the variant that triggers the issue: data type, rank, shape, component count, and any device feature or
  provider option.
- Mark as blocking: wrong results (including cache-key collisions), out-of-bounds access, crashes, hangs, WebGPU
  validation errors on a supported backend, a broken build target, and missing tests for changed behavior.
- Report performance concerns only with a concrete mechanism, such as per-run pipeline creation, a host readback, or
  lost vectorization.
- Label defects that predate the change as pre-existing, separately from findings the change introduces.
- When something could not be verified, such as code outside the diff or behavior on a specific vendor, state what is
  unverified and the check that would settle it instead of guessing.
