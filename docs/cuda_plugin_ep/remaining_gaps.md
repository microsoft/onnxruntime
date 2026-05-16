# CUDA Plugin EP — Remaining Gaps

This document tracks known gaps between the CUDA plugin EP and the bundled CUDA EP.
It is derived from Section 15 (Future Work) of `cuda_plugin_ep_design.md` and reflects
the state as of May 2026.

Most standard and contrib operators are
auto-registered via `PluginKernelCollector`; the gaps below are the explicitly excluded
categories listed in `cmake/onnxruntime_providers_cuda_plugin.cmake`.

---

## 1. Excluded Operator Categories

The following source trees are filtered from the plugin build. Each exclusion has a
specific technical reason and a path to re-inclusion.

| Excluded path | Reason | Re-inclusion path |
|---|---|---|
| `contrib_ops/cuda/llm/*` | Needs a dedicated adapter pass | Resolve plugin build failures under `ORT_USE_EP_API_ADAPTERS`, then remove CMake filter and add tests |
| `contrib_ops/cuda/transformers/*` | Beam search / greedy search / sampling require subgraph inference not yet exposed via `OrtEp` API | Requires new `OrtEp` API surface for subgraph-level execution |
| `contrib_ops/cuda/collective/*` | NCCL multi-GPU communication ops — intentionally out of scope | Out of scope for standalone plugin |
| `contrib_ops/cuda/aten_ops/*` | ATen/PyTorch tensor interop — intentionally out of scope | Out of scope for standalone plugin |
| `contrib_ops/cuda/tensor/shrunken_gather.cc` | Training op; `provider_api.h` in header conflicts with plugin build | Fix header dependency or add adapter guard |
| `tensor/sequence_op.cc` | Uses `TensorSeq` which is an incomplete type in the plugin build | Complete the `TensorSeq` C++ adapter in `include/onnxruntime/ep/adapter/` |
| `tensor/size.cc`, `tensor/shape_op.cc` | Pure CPU ops; `OpKernel` base class cannot convert to `ep::adapter::OpKernel` | Permanent — handled by `GetCpuPreferredNodes` |
| `tunable/*` | Depends on `CudaTuningContext` and other framework-only CUDA EP infrastructure | Implement a plugin-side `ITuningContext`; see §3 below |
| `controlflow/if.cc`, `loop.cc`, `scan.cc` | Framework versions inherit CPU base classes unavailable in plugin build | Replaced by `cuda_controlflow_plugin.cc`; framework versions remain excluded |

**Note:** Standard ONNX ops, NHWC kernels, and most contrib ops outside the categories
above are included in the plugin build and auto-register via `PluginKernelCollector`.

---

## 2. Provider Options Gaps

The following `OrtCUDAProviderOptionsV2` / session option keys supported by the bundled
EP are not yet wired up in the plugin EP's `CudaEpFactory::ParseConfig()`.

| Option key | Notes |
|---|---|
| `tunable_op_enable`, `tunable_op_tuning_enable`, `tunable_op_max_tuning_duration_ms` | Blocked on tunable ops framework (§3) |
| `has_user_compute_stream` / `user_compute_stream` | Plugin `OrtEpCallbacks` has no callback to receive an external stream pointer |
| `do_copy_in_default_stream` | Framework-internal scheduling option; needs exposure via `OrtEp` API |
| `use_ep_level_unified_stream` | Framework-internal; needs exposure via `OrtEp` API |
| `external_allocator_info` (alloc/free/empty_cache function pointers) | No equivalent in the plugin allocator interface |

---

## 3. Tunable Ops

**Status:** Open.

The tunable MatMul/Gemm variants in `tunable/math/matmul.cc` and `tunable/math/gemm.cc`
are guarded by `#ifndef ORT_USE_EP_API_ADAPTERS` and fall back to non-tuned paths in
the plugin build. The entire `tunable/` subtree is excluded from the plugin because it
depends on `CudaTuningContext`.

**Required work:** Implement a plugin-side `ITuningContext` that exposes the
enable/disable and profiling controls through session options, remove the
`ORT_USE_EP_API_ADAPTERS` guards in `matmul.cc`/`gemm.cc`, and remove the `tunable/*`
CMake exclusion filter. Add corresponding session option parsing (§2 above) and tests.

---

## 4. TensorSeq Adapter

**Status:** Open. Blocks `sequence_op.cc`.

`identity_op.cc` is already included (its `TensorSeq` code path is guarded by
`#ifndef BUILD_CUDA_EP_AS_PLUGIN`). `sequence_op.cc` is the last file still excluded
due to `TensorSeq` being an incomplete type in the plugin build.

**Required work:** Extend the C++ adapter layer in
`include/onnxruntime/ep/adapter/` to expose enough of `TensorSeq` to compile
`sequence_op.cc`. Also extend the ORT C API for remaining framework-style attribute
accessors (e.g., string-array attributes used by RNN kernels) if needed.

---

## 5. OpSchema-Validated Kernel Registration

**Status:** Open. `OrtEpApi` schema query wrappers are available (landed in PR #27713).

Four sub-tasks remain:

**A. Registration-time validation pass** *(highest value, lowest risk)*
Add a debug/diagnostic loop in `CreateCudaKernelRegistry()` (`cuda_plugin_kernels.cu`)
that validates every registered kernel's type-constraint names against the ONNX schema.
Catches silent constraint-name drift without changing the registration flow.

**B. NHWC internal-domain schema diagnostics**
Extend the validation pass (A) to cover `com.ms.internal.nhwc`-domain registrations.
When a kernel lookup fails for an NHWC-rewritten node, emit the expected vs. registered
constraint name — directly addressing the diagnostic gap in §5.3.1.3 of the design doc.

**C. Parity report enhancement**
Update `tools/ci_build/cuda_plugin_parity_report.py` to use the schema API to flag
type-constraint mismatches, in addition to the existing op-coverage comparison.

**D. Schema-driven `KernelDefBuilder` helpers** *(longer term)*
Create a `KernelDefBuilder` helper that auto-derives constraint names from the schema
instead of requiring hard-coded strings. Lower priority than A–C.

---

## 6. NHWC Cleanup and Hardening

**Status:** Partially complete. Runtime NHWC callbacks, second-pass capability handling
for pre-assigned NHWC nodes, and focused Conv/BatchNorm/Pool tests are in place.

Remaining work:

- **Unified allowlist:** Reconcile the NHWC conversion allowlist in the plugin with
  `cuda_nhwc_kernels.cc` (excluded from the plugin build). Currently the plugin's NHWC
  support is ad-hoc; gaps in coverage vs. the bundled EP are not surfaced.
- **Kernel-miss diagnostics:** When a node in `com.ms.internal.nhwc` domain fails lookup
  (no matching plugin kernel), the current diagnostics do not report which op/version/type
  was missed. Blocked on completing task 5B above.
- **Structural assertions:** Add test-time assertions that plugin-backed NHWC kernels
  were actually selected after a session with `prefer_nhwc=true`, rather than silently
  falling back to NCHW or CPU.
- **Expanded test coverage:** Current NHWC tests cover ~4 ops (Conv, BatchNormalization,
  MaxPool, AveragePool). The bundled EP supports ~20+ NHWC kernels. Expand coverage as
  the unified allowlist is established.

---

## 7. Annotation-Based Partitioning — `gpu:<index>()` Selector

**Status:** Partially resolved in commit f7113bdc (#28028).

The `cuda(...)` device selector and general `gpu:nvidia` matching now include
`CudaPluginExecutionProvider` alongside `CUDAExecutionProvider`. The remaining known
limitation:

- **`gpu:<index>(...)` ordinal reliability** — `OrtHardwareDevice::device_id` is not
  guaranteed to be a stable CUDA ordinal, making index-based layer assignment unreliable
  when multiple similar NVIDIA GPUs are present on the same host. Reliable multi-GPU
  annotation requires using the CUDA device ordinal as the canonical identifier.

---

## 8. CI — Performance Gates and Feature-Specific Coverage

**Status:** Partially complete. Linux and Windows plugin CI workflows
(`linux_cuda_plugin_ci.yml`, `windows_cuda_plugin.yml`) build and run
`test_cuda_plugin_ep.py`.

Remaining work:

- Add targeted benchmarks or perf gates for CUDA graph replay and allocator behavior.
- Extend CI test coverage once profiling improvements (§9 below) and tunable-op support
  (§3 above) land.
- Add CI coverage for contrib LLM kernels once `contrib_ops/cuda/llm/*` is re-included.

---

## 9. Profiling — Per-Node Attribution and GPU→ORT Event Linkage

**Status:** CUPTI-based GPU activity tracing, external correlation ID mapping, and event
collection via `OrtEpProfilerImpl` are implemented.

Known limitations:

- **GPU→ORT event linkage is implicit.** GPU events carry CUPTI metadata
  (`stream`, `grid_*`, `block_*`) but no ORT correlation or parent identifier. Downstream
  consumers must relate GPU kernels to ORT nodes via timestamp proximity. Future work may
  attach `correlation_id` or parent event name via `StopEvent`'s `OrtProfilingEvent`
  parameter.
- **No per-node profiling attribution.** There is no mechanism to attribute GPU kernel
  time back to a specific ORT graph node in the plugin EP profiling output.

---

## 10. Packaging and Publishing

**Status:** Build scripts exist; no automatic publish pipeline.

| Artifact | Status |
|---|---|
| Python wheel (`onnxruntime-ep-cuda12/13`) | Build script at `plugin-ep-cuda/python/build_wheel.py`; wheels built in CI but not published automatically |
| NuGet (C#) | Build script at `plugin-ep-cuda/csharp/pack_nuget.py`; not published automatically |
| Standalone prebuilt binaries | Not available; users must build from source |
| Deployment | Manual: copy the `.so` / `.dll` into the ORT package's `capi/` directory |

**Required work:** Wire up automatic publish steps in the CI workflows for both PyPI
(Python wheel) and NuGet after a successful build-and-test run, gated on a version tag.

---

## Summary Table

| Area | Status |
|---|---|
| Standard + contrib ops (non-excluded) | Included via auto-registration |
| Contrib LLM (`llm/*`) | Excluded — adapter pass needed |
| Transformers (`transformers/*`) | Excluded — subgraph API needed |
| Collective/NCCL, ATen | Excluded — intentionally out of scope |
| `sequence_op.cc` | Excluded — TensorSeq adapter incomplete |
| Tunable ops | Excluded — `ITuningContext` not yet implemented |
| Provider options (user stream, external allocator, unified stream, do_copy, tunable) | Not implemented |
| OpSchema validation (A–D) | Open |
| NHWC cleanup (allowlist, diagnostics, assertions) | Partially complete |
| `gpu:<index>()` selector reliability | Open |
| CI perf gates and extended coverage | Partially complete |
| Profiling per-node attribution | Open |
| Automatic PyPI / NuGet publishing | Open |
| Resource accounting | Done (f7113bdc) |
| `cuda(...)` / `gpu:nvidia` device selector for plugin EP | Done (f7113bdc) |
