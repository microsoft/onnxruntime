# CUDA Plugin EP — Design Document

## 1. Overview

The CUDA Plugin EP is an alternative build of the ONNX Runtime CUDA Execution Provider that compiles as a standalone shared library (`libonnxruntime_providers_cuda_plugin.so`). It loads at runtime through the ORT EP Plugin API instead of being statically linked into the main runtime binary.

**Goals:**
- Allow CUDA EP updates independent of ORT core releases
- Support all operators currently supported by the in-tree CUDA EP (tunable ops are low priority)
- Minimize changes to existing CUDA kernel source files

**Current status:** ~80% of CUDA kernels compile in the plugin build. Excluded operators are documented in [Section 7](#7-excluded-operators).

---

## 2. Architecture

### 2.1 Build Targets

The ORT CUDA build produces four separate libraries:

| Target | Output | Type | Description |
|--------|--------|------|-------------|
| `onnxruntime_providers` | `libonnxruntime_providers.a` | Static lib | CPU provider + framework ops |
| `onnxruntime_providers_shared` | `libonnxruntime_providers_shared.so` | Shared lib | DLL-boundary bridge for in-tree EPs |
| `onnxruntime_providers_cuda` | `libonnxruntime_providers_cuda.so` | Shared module | In-tree CUDA EP (uses `SHARED_PROVIDER` bridge) |
| `onnxruntime_providers_cuda_plugin` | `libonnxruntime_providers_cuda_plugin.so` | Shared module | Plugin CUDA EP (uses EP API adapters) |

### 2.2 Preprocessor Defines

Each build target uses different preprocessor defines that control how framework types are resolved:

| Define | Set In | Purpose |
|--------|--------|---------|
| `SHARED_PROVIDER` | `onnxruntime_providers_shared`, `onnxruntime_providers_cuda` | Activates the DLL-boundary proxy types in `provider_api.h` |
| `BUILD_CUDA_EP_AS_PLUGIN` | `onnxruntime_providers_cuda_plugin` | Makes `provider_api.h` a no-op; activates plugin-specific code paths |
| `ORT_USE_EP_API_ADAPTERS` | `onnxruntime_providers_cuda_plugin` | Enables the EP adapter type aliases (`ep/adapters.h`) |
| `ORT_API_MANUAL_INIT` | `onnxruntime_providers_cuda_plugin` | Manual ORT API initialization in plugin DLL |

### 2.3 Class Hierarchy

```
OrtEpFactory                              OrtEp
    ↑                                       ↑
CudaEpFactory                     adapter::Ep (holds unique_ptr<IExecutionProvider>)
    │                                        ↑
    │                                     CudaEp
    │                                        │
    │                                        └─ owns ──→ CUDAExecutionProvider
    │                                                     (: IExecutionProvider)
    │                                                     ├─ config members
    │                                                     ├─ device properties
    │                                                     └─ stream→handle map
    │
    └─ creates ──→ CudaSyncStream (owns cublasHandle_t, cudnnHandle_t, cublasLtHandle_t)
```

Key ownership relationships:
- `CudaEpFactory` creates `CudaEp` instances and `CudaSyncStream` objects
- `CudaEp` inherits from `ep::adapter::Ep` and owns a `CUDAExecutionProvider` instance (accessible via `EpImpl()`)
- `CUDAExecutionProvider` is a plugin-local class (not the framework one) that inherits from `IExecutionProvider` and provides the full API surface CUDA kernels need
- `CudaSyncStream` owns CUDA/cuBLAS/cuDNN handles per stream

### 2.4 Plugin DLL Entry Points

The plugin exports exactly two C symbols:
- `CreateEpFactories()` — called by ORT to create the EP factory
- `ReleaseEpFactory()` — called by ORT to destroy the factory

All other symbols have hidden visibility.

---

## 3. Type Resolution — How Kernel Code Compiles Unchanged

The core design principle is that existing CUDA kernel `.cc` files compile in the plugin build with **zero or minimal source changes**. This is achieved through a two-layer force-include mechanism.

### 3.1 Force-Include Chain

For every `.cc` file in the plugin build, CMake injects two force-includes before any source code:

```
1. ep/adapters.h         — adapter type aliasing
2. cuda_kernel_adapter.h — CudaKernel base class, macros, CPU shims
```

Note: `.cu` files do NOT receive force-includes (conflicts with CUTLASS/cute). They must include `cuda_kernel_adapter.h` explicitly if needed.

### 3.2 Adapter Type Aliasing (`ep/adapters.h`)

`ep/adapters.h` defines `using` aliases in both `onnxruntime::cuda` and `onnxruntime::contrib::cuda` namespaces:

```cpp
namespace onnxruntime::cuda {
    using OpKernel       = ep::adapter::OpKernel;
    using OpKernelContext = ep::adapter::OpKernelContext;
    using OpKernelInfo   = ep::adapter::OpKernelInfo;
    using KernelRegistry = ep::adapter::KernelRegistry;
    using KernelDefBuilder = ep::adapter::KernelDefBuilder;
    using DataTransferManager = ep::adapter::DataTransferManager;
    // ... etc
}
```

When kernel code in `namespace onnxruntime::cuda` references `OpKernelContext`, it resolves to the adapter type instead of the framework type. **No kernel source changes needed.**

### 3.3 Provider API Bypass

In the plugin build, `provider_api.h` (normally included from `cuda_common.h`) is a **complete no-op** — it does NOT define `SHARED_PROVIDER`. This means:

- `#ifndef SHARED_PROVIDER` guards in framework headers remain **active**, exposing real types
- Header-inlined utility methods (see [Section 4](#4-cpu-base-class-helpers)) get their inline bodies
- The `ProviderHostCPU` virtual table bridge is bypassed entirely

### 3.4 Kernel Adapter (`cuda_kernel_adapter.h`)

This 700+ line header provides everything CUDA kernels need that would normally come from framework infrastructure:

| Section | What It Provides |
|---------|-----------------|
| Error macros | `CUDA_RETURN_IF_ERROR`, `CUBLAS_RETURN_IF_ERROR`, `CUDNN_RETURN_IF_ERROR` |
| Type mappings | `ToCudaType<MLFloat16>::MappedType = half`, etc. |
| CudaKernel base | Scratch buffers, handle access, `Stream()`, `GetComputeStream()` |
| Kernel registration | Self-registering `ONNX_OPERATOR_*_KERNEL_EX` macro overrides via `PluginKernelCollector` |
| CPU shims | Lightweight reimplementations of CPU helpers not linked into plugin |
| Math helpers | `HalfGemmOptions`, `CublasMathModeSetter` |
| Stream shim | `PluginStreamShim` wrapping raw `cudaStream_t` as `onnxruntime::Stream*` |

### 3.5 Kernel Registration

In the in-tree build, kernels register through centralized tables (`cuda_nhwc_kernels.cc`, `cuda_contrib_kernels.cc`). In the plugin build, the `ONNX_OPERATOR_*_KERNEL_EX` macros are overridden to auto-register each kernel into the `PluginKernelCollector` singleton at static initialization time:

```cpp
// Macro override generates:
// 1. BuildKernelCreateInfo<CLASS>() function
// 2. Static PluginKernelCollector::Register() call

// At plugin startup, CreateCudaKernelRegistry() iterates the collector
// and registers each kernel into an adapter::KernelRegistry.
```

---

## 4. CPU Base Class Helpers — The SHARED_PROVIDER Pattern

Many CUDA kernels inherit from CPU base classes and call utility methods (e.g., `PadBase::HandleDimValueZero`, `SliceBase::PrepareForCompute`). In the in-tree build, these call across the DLL boundary through `ProviderHostCPU`. The plugin doesn't use this bridge.

### 4.1 Pattern: Inline in Header

The primary approach moves pure-computation helpers from CPU `.cc` files to headers:

```cpp
// In padbase.h:
#ifdef SHARED_PROVIDER
  // In-tree build: declaration only, body in ProviderHostCPU bridge
  static void HandleDimValueZero(Mode mode, const TensorShape& input_shape, TensorShape& output_shape);
#else
  // Plugin build + CPU provider: inline body
  static inline void HandleDimValueZero(Mode mode, const TensorShape& input_shape,
                                        TensorShape& output_shape) {
    // ... implementation ...
  }
#endif
```

**Files refactored with this pattern:**
- `padbase.h` — `HandleDimValueZero`, `ComputePads` (delegates to `ComputePadsImpl` template)
- `scatter_nd.h` — `ValidateShapes`
- `split.h` — `PrepareForCompute`
- `tile.h` — `IsTileMemcpy`
- `slice.h` — `PrepareForCompute`, `FlattenOutputDims`
- `cumsum.h` — `cumsum_op::GetAxis`
- `bias_gelu_helper.h` — `bias_gelu_helper::CheckInputs`
- `concatbase.h` — `PrepareForCompute`
- `gatherbase.h` — `PrepareForCompute`/`PrepareForComputeImpl` (template)
- `unsqueeze.h` — `PrepareCompute`
- `embed_layer_norm_helper.h` — `embed_layer_norm::CheckInputs` (templatized on context type)
- `non_max_suppression_helper.h` — `NonMaxSuppressionBaseImpl` template class (new file)
- `attention_base.h` — `AttentionBase::CheckInputs`, `CheckMask`, `GetPresent` (templatized on context type)
- `longformer_attention_base.h` — `LongformerAttentionBase::CheckInputs`
- `roialign.h` — `CheckROIAlignValidInput`, `RoiAlignBase` constructor (templatized on info type)
- `upsamplebase.h` — `UpsampleBase::AdjustOutputSizeAsPolicy`
- `crop.h` — `CropBase` constructor (templatized on info type)
- `space_depth_ops.h` — `SpaceDepthBase` constructor (templatized on info type)
- `clip.h` — Clip min/max attribute handling (removed `Clip_6Base` CPU dependency)
- `cuda_common_type_helpers.h` — CUDA type conversion and handle error string helpers (moved from `cuda_common.cc`)

### 4.2 Pattern: Template Methods

For methods that take `OpKernelContext&` (which differs between plugin and in-tree builds), template versions accept any context type:

```cpp
// In padbase.h:
template <typename KernelContextType>
static void ComputePadsImpl(KernelContextType& ctx, size_t data_rank,
                            gsl::span<const int64_t> pads_data, PadsVector& pads) { ... }
```

The CUDA kernel calls `PadBase::ComputePadsImpl(*ctx, ...)` directly, avoiding the `OpKernelContext&` type mismatch.

The same pattern is applied to constructors that receive `OpKernelInfo`:

```cpp
// In roialign.h:
template <typename TKernelInfo>
RoiAlignBase(const TKernelInfo& info) {
  info.template GetAttr<std::string>("mode", &mode_string);
  info.template GetAttr<int64_t>("output_height", &output_height_);
  // ...
}
```

This allows the base class constructor to work with both the framework `OpKernelInfo` and the plugin adapter's `OpKernelInfo`. Applied to: `RoiAlignBase`, `CropBase`, `SpaceDepthBase` (#27628).

### 4.3 Files That Cannot Be Inlined

Some CPU base classes have heavy dependencies (protobuf, `UnpackTensor`) that make inlining impractical:

- **`ConstantOfShapeBase`** — depends on `TensorProto` and `UnpackTensor`. Plugin uses a self-contained duplicate class in `constant_of_shape.h` guarded by `#ifdef BUILD_CUDA_EP_AS_PLUGIN`.
- **`UpsampleBase`** — partially addressed: `AdjustOutputSizeAsPolicy` moved to header (#27628). Still depends on `InputDefs()` and `OpKernelInfo::GetAllocator()` which are not in the adapter.

---

## 5. Handle and Stream Management

### 5.1 Stream Ownership

`CudaSyncStream` is the plugin's CUDA stream implementation:
- Owns `cudaStream_t`, `cublasHandle_t`, `cudnnHandle_t`, `cublasLtHandle_t`
- Created by `CudaEpFactory::CreateSyncStreamForDevice`
- Registered with `CUDAExecutionProvider` for handle lookup

### 5.2 Handle Access Path

```
CudaKernel::GetCublasHandle(OpKernelContext* ctx)
  → Stream(ctx)                                     // raw cudaStream_t from ctx
  → CUDAExecutionProvider::GetActiveProvider()       // static pointer to active EP
  → provider->GetCublasHandle(cudaStream_t)          // stream→handle map lookup
```

The `CUDAExecutionProvider` maintains a `std::unordered_map<cudaStream_t, CudaSyncStream*>` for handle lookups.

### 5.3 Provider Access

Kernels access the provider through two paths:
1. **`CudaKernel::provider_`** — set in the constructor from `info.GetExecutionProvider()`
2. **`CUDAExecutionProvider::GetActiveProvider()`** — static atomic pointer (for `.cu` code that doesn't have a `CudaKernel` instance)

### 5.4 CUDA Graph Support

#### 5.4.1 How CUDA Graph Works in Bundled CUDA EP

CUDA Graph capture/replay in ORT is a **cooperative protocol** between the ORT session framework and the execution provider. Understanding this protocol is critical for the plugin EP design.

**Session-level orchestration** (`inference_session.cc`):

1. During session initialization, if an EP reports `IsGraphCaptureEnabled() == true` and all graph nodes are assigned to that EP (plus allowed CPU shape nodes), the session caches a pointer to the EP in `cached_execution_provider_for_graph_replay_`.

2. At `Run()` time, the session checks `cached_execution_provider_for_graph_replay_.IsGraphCaptured(annotation_id)`:
   - **If captured**: The session **skips the entire kernel dispatch pipeline** — no `OnRunStart`, no executor, no `OnRunEnd` — and calls `ReplayGraph(annotation_id)` directly. This is the fast path.
   - **If not yet captured**: The session runs the normal kernel dispatch pipeline (including `OnRunStart` → executor → `OnRunEnd`), which allows the EP to manage warm-up counting and trigger capture.

3. After each normal run, the session checks if graph capture is enabled but not yet captured, and **recursively calls `Run()`** to accumulate the required warm-up runs and trigger capture — so from the user's perspective, a single `Run()` call handles the entire warm-up + capture sequence.

**EP-level capture** (`CUDAExecutionProvider`):

- `OnRunStart()`: If warm-up is complete and graph not yet captured, calls `cudaStreamBeginCapture()`.
- `OnRunEnd()`: If capturing, calls `cudaStreamEndCapture()` + `cudaGraphInstantiate()` + first `Replay()` (since captured kernels don't execute on GPU during capture).
- `IsGraphCaptureEnabled()`: Returns `true` if `enable_cuda_graph` provider option is set.
- `IsGraphCaptured(annotation_id)`: Returns `true` if a graph has been captured for this annotation.
- `ReplayGraph(annotation_id)`: Calls `cudaGraphLaunch()` for the stored `cudaGraphExec_t`.

The key insight is that the **session-level replay bypass** (`ReplayGraph()` without kernel dispatch) is what makes CUDA Graph efficient. Without it, the EP can capture a graph but can never replay it efficiently — kernels would still be dispatched by the executor on every run.

```
Session::Run()
  ├── [Graph captured?] ──YES──→ ep->ReplayGraph(id) ──→ return   ← FAST PATH
  │
  └── [Not captured] ──→ OnRunStart() → executor dispatches kernels → OnRunEnd()
                              │                                            │
                              │  (EP begins cudaStreamBeginCapture)         │  (EP ends capture, first replay)
                              │                                            │
                              └──────── Session recurses if warmup needed ─┘
```

#### 5.4.2 Current Plugin EP Behavior — API Gap

The `OrtEp` C API (`onnxruntime_ep_c_api.h`) provides `OnRunStart` and `OnRunEnd` callbacks but **does not include**:
- `IsGraphCaptureEnabled()`
- `IsGraphCaptured(annotation_id)`
- `ReplayGraph(annotation_id)`

The `PluginExecutionProvider` bridge (`ep_plugin_provider_interfaces.cc`) does not override these `IExecutionProvider` virtual methods, so they return the base class defaults (`false`, `false`, `Status::OK()`).

**Consequence**: The session's `cached_execution_provider_for_graph_replay_` is never set for the plugin EP. The session-level replay bypass **never activates**. Even after the plugin captures a CUDA graph via `OnRunStart`/`OnRunEnd`, subsequent runs still go through the full kernel dispatch pipeline — the captured graph sits unused.

The current plugin implementation has a partial mitigation: it captures the graph and replays it once (in `OnRunEnd` after capture). But on subsequent runs, `OnRunEnd` sees the graph is already captured and does nothing.

#### 5.4.3 Revised Design — Remove EP-Level Graph Management

Given the API gap, the correct design for the plugin EP is:

> **The plugin EP should NOT manage CUDA graph capture/replay internally.** CUDA graph support requires session-level cooperation that is not available through the current `OrtEp` C API.

**Rationale:**

1. The `OrtEp` C API has no `IsGraphCaptureEnabled`/`IsGraphCaptured`/`ReplayGraph` callbacks. Without these, the session cannot know that the EP supports graph capture, cannot bypass kernel dispatch for replay, and cannot trigger the recursive warm-up sequence.

2. Implementing capture in `OnRunStart`/`OnRunEnd` without the session-level replay bypass is **incorrect** — the captured graph would never be replayed on subsequent runs (the session always dispatches kernels normally).

3. The session's graph validation logic (all nodes on one EP, no control flow) is also not triggered without `IsGraphCaptureEnabled()`.

**Recommended approach:**

| Option | Description | Effort | Status |
|--------|------------|--------|--------|
| **A. Extend the OrtEp C API** | Add `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph` to `OrtEp`. Update `PluginExecutionProvider` to delegate to these. | Medium — requires ORT core changes | Preferred long-term solution |
| **B. Disable graph capture in plugin EP** | Remove `CUDAGraphManager` and graph-related code from the plugin. Document as a known limitation. Re-enable when Option A is available. | Small | Recommended for now |
| **C. Keep capture-only (no replay)** | Keep the current code but document that it only captures + replays once (the first time), with no subsequent replay optimization. | None | Misleading — gives false confidence |

**Recommendation**: Option B for the current release, with Option A tracked as a public API enhancement request.

#### 5.4.4 What Needs to Change in ORT Core (Option A)

To enable full CUDA graph support for plugin EPs, the `OrtEp` struct needs three new optional callbacks:

```c
// Proposed additions to OrtEp (onnxruntime_ep_c_api.h)
struct OrtEp {
  // ... existing fields ...

  /// Returns true if CUDA graph capture is enabled for this EP.
  /// If nullptr, defaults to false.
  ORT_API2_STATUS(IsGraphCaptureEnabled, _In_ const OrtEp* this_ptr, _Out_ bool* enabled);

  /// Returns true if a graph has been captured for the given annotation ID.
  /// If nullptr, defaults to false.
  ORT_API2_STATUS(IsGraphCaptured, _In_ const OrtEp* this_ptr,
                  _In_ int graph_annotation_id, _Out_ bool* captured);

  /// Replay a previously captured graph.
  /// If nullptr, returns OK (no-op).
  ORT_API2_STATUS(ReplayGraph, _In_ OrtEp* this_ptr, _In_ int graph_annotation_id);
};
```

The `PluginExecutionProvider` bridge would then delegate these to the plugin:

```cpp
// In ep_plugin_provider_interfaces.cc
bool PluginExecutionProvider::IsGraphCaptureEnabled() const {
  if (ort_ep_->IsGraphCaptureEnabled == nullptr) return false;
  bool enabled = false;
  auto* status = ort_ep_->IsGraphCaptureEnabled(ort_ep_.get(), &enabled);
  // handle status...
  return enabled;
}
```

This would plug into the existing `cached_execution_provider_for_graph_replay_` mechanism in `InferenceSession` with no other session-level changes needed.

#### 5.4.5 Current State

| Component | Status | Notes |
|-----------|--------|-------|
| `cuda_graph_plugin.h/.cc` | Implemented | `CUDAGraphManager` adapted from bundled EP. Captures/replays correctly. |
| `CudaEp::OnRunStartImpl` | Implemented | Reads `gpu_graph_id`, manages warm-up, begins capture. |
| `CudaEp::OnRunEndImpl` | Implemented | Ends capture, first replay. No subsequent replay. |
| Session-level replay bypass | **Not functional** | `OrtEp` API lacks `IsGraphCaptureEnabled`/`IsGraphCaptured`/`ReplayGraph`. |
| Tests | Pass (capture + first replay) | `test_cuda_plugin_cuda_graph()` tests warm-up, capture, and `gpu_graph_id=-1` disable. |

**Action items:**
1. Keep `cuda_graph_plugin.h/.cc` and `CudaEp` graph state machine code — it is correct and will be needed when the API gap is closed.
2. Default `enable_cuda_graph` to `false` in the plugin EP config and document the limitation.
3. File an ORT core feature request to add `IsGraphCaptureEnabled`/`IsGraphCaptured`/`ReplayGraph` to the `OrtEp` C API.
4. When the API is extended, wire up the existing `CUDAGraphManager` through the new callbacks.

---

## 6. EP Adapter Layer (`include/onnxruntime/ep/adapter/`)

The adapter layer provides thin wrappers around the ORT C API that present a C++ interface matching the framework types:

| Adapter Class | Wraps | Key Methods |
|---------------|-------|-------------|
| `OpKernel` | `OrtKernelImpl` | `Compute()`, `PrePack()` |
| `OpKernelContext` | `OrtKernelContext` | `Input<Tensor>()`, `Output()`, `InputCount()`, `GetGPUComputeStream()`, `GetComputeStream()` |
| `OpKernelInfo` | `Ort::ConstKernelInfo` | `GetAttr<T>()`, `GetExecutionProvider()`, `TryGetConstantInput()`, `GetDataTransferManager()` |
| `KernelRegistry` | `Ort::KernelRegistry` | `Register(KernelCreateInfo&&)` |
| `KernelDefBuilder` | `Ort::KernelDefBuilder` | `TypeConstraint()`, `InputMemoryType()`, `SetName()` |
| `Ep` | `OrtEp` | `EpImpl()`, allocators, data transfer |
| `Logger` | Plugin logger | Logging interface |
| `DataTransferManager` | `IDataTransfer` | `CopyTensor()` |

---

## 7. Excluded Operators

The following operators are excluded from the plugin build. Each exclusion has a specific technical reason and a path to inclusion.

### 7.1 Infrastructure (Permanently Excluded — Replaced by Plugin Equivalents)

| File | Reason |
|------|--------|
| `cuda_execution_provider.cc` | Replaced by `cuda_ep_provider.h` + `cuda_ep.h` |
| `cuda_provider_factory.cc` | Replaced by `cuda_ep_factory.cc` |
| `cuda_provider_interface.cc` | Not needed in plugin architecture |
| `cuda_stream_handle.cc` | Replaced by `cuda_stream_plugin.cc` |
| `cuda_execution_provider_info.cc` | Config parsed directly in `CudaEp::Config` |
| `cuda_graph.cc` | Replaced by `cuda_graph_plugin.cc` |
| `cuda_mempool_arena.cc` | Plugin uses `cudaMalloc`/`cudaFree` directly |
| `cuda_common.cc` | Utility functions shimmed in `cuda_kernel_adapter.h` |
| `cuda_nhwc_kernels.cc` | Replaced by `PluginKernelCollector` auto-registration |
| `cuda_contrib_kernels.cc` | Replaced by `PluginKernelCollector` auto-registration |

### 7.2 Pure CPU Ops (Permanently Excluded)

| File | Reason |
|------|--------|
| `tensor/size.cc` | Pure CPU op, handled by `GetCpuPreferredNodes` |
| `tensor/shape_op.cc` | Pure CPU op, inherits from `onnxruntime::OpKernel` (framework) |

### 7.3 Operators Excluded Due to Missing Features

| File | Exclusion Reason | What's Needed to Include |
|------|-----------------|--------------------------|
| `controlflow/*` | CPU base class If/Loop/Scan not linked | Plugin has own wrappers in `cuda_controlflow_plugin.cc` via `OrtEpApi`. Already functional. |
| `tunable/*` | Depends on real `CudaTuningContext` | Implement plugin-side `ITuningContext` that delegates to ORT tuning APIs. Low priority. |
| `rnn/*` | ORT C API lacks `KernelInfoGetAttributeArray_string` | Extend C API with string-array attribute support. |
| `math/einsum.cc`, `math/einsum_utils/*` | `einsum_auxiliary_ops.cc` calls `ReductionOps::ReduceCompute` (framework-only) | Extract `ReduceCompute` into a shared interface or reimplement the reduction path. |
| `tensor/identity_op.cc` | Uses `TensorSeq` (incomplete type in plugin) | Add `TensorSeq` adapter to the EP adapter layer. |
| `tensor/sequence_op.cc` | Uses `TensorSeq` (incomplete type in plugin) | Same as above. |
| `tensor/space_depth_ops.cc` | Inherits `SpaceDepthBase` (CPU provider) | Constructor templatized on `KernelInfoType` (#27628). Remaining: inline `SpaceDepthCompute` validation logic. |
| `tensor/upsample.cc` | `UpsampleBase` uses `InputDefs()` and `OpKernelInfo::GetAllocator()` | `AdjustOutputSizeAsPolicy` moved to header (#27628). Remaining: extend adapter with `GetAllocator()` and `InputDefs()`. |
| `tensor/resize.cc` | Inherits from `Upsample` (excluded above) | Fix `Upsample` first, then `Resize` follows. |
| `generator/constant_of_shape.cc` | `ConstantOfShapeBase` depends on `TensorProto`/`UnpackTensor` | Plugin already has self-contained implementation in `constant_of_shape.h` via `#ifdef BUILD_CUDA_EP_AS_PLUGIN`. The `.cc` is excluded but the kernel works. |
| `object_detection/*` | `NonMaxSuppressionBase`, `RoiAlignBase` from CPU provider | `NonMaxSuppressionBaseImpl` template (#27617), `RoiAlignBase` constructor templatized (#27628). Remaining: integration verification. |
| `llm/*` | Attention ops dereference `onnxruntime::Stream*` (not adapter-compatible) | Extend adapter `OpKernelContext::GetComputeStream()` to return a full `Stream*` implementation. |
| `contrib_ops/cuda/llm/*` | Same as above | Same as above. |
| `contrib_ops/cuda/bert/attention.cc` | `GetComputeStream()` returns real `Stream*` which is needed | `AttentionBase` helpers moved to header (#27628). Remaining: `Stream*` adapter extension for `QkvToContext`. |
| `contrib_ops/cuda/bert/decoder_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/decoder_masked_self_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/embed_layer_norm.cc` | `EmbedLayerNormHelper` CPU base class | Already refactored helper; needs `GetComputeStream()` fix. |
| `contrib_ops/cuda/bert/fast_gelu.cc` | Was excluded due to `bias_gelu_helper` CPU base class dep | `bias_gelu_helper::CheckInputs` is now inlined. Remove this exclusion and verify. |
| `contrib_ops/cuda/bert/group_query_attention.cc` | `GetComputeStream()` / attention infra | Same `Stream*` adapter extension. |
| `contrib_ops/cuda/bert/longformer_attention.cc` | `LongformerAttentionBase::CheckInputs` moved to header (#27628) | `Stream*` adapter extension. |
| `contrib_ops/cuda/bert/multihead_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/packed_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/packed_multihead_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/paged_attention.cc` | Same | Same. |
| `contrib_ops/cuda/bert/relative_attn_bias.cc` | Same | Same. |
| `contrib_ops/cuda/bert/remove_padding.cc` | Same | Same. |
| `contrib_ops/cuda/diffusion/group_norm.cc` | `GetComputeStream()` | Same `Stream*` adapter extension. |
| `contrib_ops/cuda/fused_conv.cc` | Framework type deps | Audit specific deps; likely `Stream*` related. |
| `contrib_ops/cuda/inverse.cc` | Framework type deps | Audit specific deps. |
| `contrib_ops/cuda/math/bias_dropout.cc` | `GetComputeStream()` | Same `Stream*` adapter extension. |
| `contrib_ops/cuda/math/fft_ops.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/math/gemm_float8.cc`/`.cu` | `GetComputeStream()` in `.cu` file | Same, plus NVCC compatibility. |
| `contrib_ops/cuda/moe/moe.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/sparse/sparse_attention.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/tensor/crop.cc` | `CropBase` constructor templatized (#27628). No `GetComputeStream()` usage. | Verify compilation — very low effort. |
| `contrib_ops/cuda/tensor/dynamic_time_warping.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/tensor/dynamicslice.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/tensor/shrunken_gather.cc` | Training op, `provider_api.h` header dep | Low priority (training). |
| `contrib_ops/cuda/quantization/attention_quantization.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/quantization/matmul_bnb4.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/quantization/matmul_nbits.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/quantization/moe_quantization.cc` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/quantization/qordered_ops/*` | `GetComputeStream()` | Same. |
| `contrib_ops/cuda/transformers/*` | Beam search, greedy search, sampling | Complex framework deps; needs significant adapter work. |
| `aten_ops/*` | ATen interop | Out of scope for plugin. |
| `collective/*` | NCCL collective ops | Out of scope for plugin. |

### 7.4 Common Exclusion Themes

The majority of excluded operators fall into a few categories:

1. **`GetComputeStream()` returning `onnxruntime::Stream*`** (~25 ops) — The adapter's `GetComputeStream()` returns a `PluginCudaComputeStreamShim` which wraps a raw `cudaStream_t`. Many attention/LLM ops dereference `Stream*` expecting a `CudaStream` with extra members. **Unblocking this is the single highest-impact change.**

2. **CPU base class inheritance** (~5 ops) — Some ops inherit from CPU base classes not linked into the plugin. Most have been refactored with the inline-header pattern. `SpaceDepthBase` and `RoiAlignBase` constructors are now templatized (#27628); `NonMaxSuppressionBase` refactored to a template (#27617); `UpsampleBase::AdjustOutputSizeAsPolicy` moved to header (#27628). Remaining: `UpsampleBase` `InputDefs()`/`GetAllocator()`.

3. **Missing C API features** (~2 ops) — RNN ops need string-array attribute support via the C API.

4. **Framework-only code paths** (~3 ops) — Einsum's reduction path, tunable infrastructure.

---

## 8. Remaining `#ifdef` Guards in Kernel Code

After refactoring, only 6 files contain `BUILD_CUDA_EP_AS_PLUGIN` or `ORT_USE_EP_API_ADAPTERS` guards:

| File | Guard | Purpose | Removable? |
|------|-------|---------|------------|
| `cuda_kernel.h` | Both | Three-way gate: plugin → adapter; in-tree → real CudaKernel | No — infrastructure |
| `cuda_common.h` | Both | Logging macros, error macros, `HalfGemmOptions` | No — infrastructure |
| `cuda_execution_provider.h` | `ORT_USE_EP_API_ADAPTERS` | Skip entire class in plugin build | No — infrastructure |
| `generator/constant_of_shape.h` | `BUILD_CUDA_EP_AS_PLUGIN` | Self-contained plugin implementation | No — can't inline `ConstantOfShapeBase` |
| `math/matmul.cc` | `ORT_USE_EP_API_ADAPTERS` | Guards `FuncManager` registration (tunable) | Only when tunable is supported |
| `math/gemm.cc` | `ORT_USE_EP_API_ADAPTERS` | Guards `FuncManager` registration (tunable) | Only when tunable is supported |

All kernel-level `#ifdef` guards in operator `.cc` files have been eliminated through the inline-header refactoring pattern, except for `matmul.cc`, `gemm.cc` (tunable dispatch), and `constant_of_shape.h` (protobuf dependency).

---

## 9. Building

### 9.1 CMake Flag

The plugin is enabled by setting `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`:

```bash
sh build.sh --config Release --build_dir build/cuda --parallel --use_cuda \
    --cuda_version 12.8 --cuda_home /path/to/cuda \
    --cudnn_home /path/to/cudnn \
    --build_wheel --skip_tests \
    --cmake_generator Ninja \
    --enable_cuda_nhwc_ops \
    --cmake_extra_defines onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="90"
```

Or using the existing `cuda.sh` convenience script:

```bash
./cuda.sh --build --test_plugin   # --test_plugin sets BUILD_CUDA_EP_AS_PLUGIN=ON
```

### 9.2 Impact on Other Build Targets

The `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON` flag has **no impact** on `libonnxruntime_providers_cuda.so` or `libonnxruntime_providers_shared.so`. It only:

1. Adds the `onnxruntime_providers_cuda_plugin` target (producing `libonnxruntime_providers_cuda_plugin.so`)
2. Appends `"cuda-plugin-ep=1"` to the build info string (cosmetic)

The in-tree CUDA EP and shared provider bridge are compiled identically regardless of this flag. A single build with the flag ON produces all four libraries — there is no need for separate build scripts or build directories.

### 9.3 Plugin Independence

`libonnxruntime_providers_cuda_plugin.so` is **fully self-contained**. It does not depend on `libonnxruntime_providers_cuda.so` or `libonnxruntime_providers_shared.so` at load time. It statically links against `onnxruntime_framework`, `onnxruntime_graph`, `onnxruntime_common`, `onnxruntime_mlas`, `onnxruntime_flatbuffers`, and links dynamically against CUDA/cuDNN/protobuf. Communication with the ORT runtime happens exclusively through the C API (`OrtApi`/`OrtEpApi`) passed at load time.

### 9.4 Build Outputs

After a successful build with the plugin flag ON, `build/cuda/Release/` contains:

| File | Description |
|------|-------------|
| `libonnxruntime_providers.a` | CPU provider (static, linked into main binary) |
| `libonnxruntime_providers_shared.so` | Shared provider bridge (for in-tree CUDA EP) |
| `libonnxruntime_providers_cuda.so` | In-tree CUDA EP (uses shared provider bridge) |
| `libonnxruntime_providers_cuda_plugin.so` | Plugin CUDA EP (standalone, uses C API) |

### 9.5 Deployment

To use the plugin EP, copy the `.so` to the ORT Python package's `capi/` directory:

```bash
cp build/cuda/Release/libonnxruntime_providers_cuda_plugin.so \
   $(python -c "import onnxruntime; print(onnxruntime.__path__[0])")/capi/
```

The plugin is then available as `CudaPluginExecutionProvider` in session provider lists.

---

## 10. Testing

### 10.1 Test Script

`onnxruntime/test/python/transformers/test_cuda_plugin_ep.py` provides multi-stage testing:

| Stage | What It Tests |
|-------|---------------|
| Stage 2 | Basic ops: Add, MatMul, Gemm, Conv |
| Stage 3 | NHWC layout: Conv, BatchNorm, MaxPool, AveragePool |
| Stage 4 | CUDA Graph capture/replay |
| Stage 5A | Standard ops: Reshape, Split, Concat, Gather, Unsqueeze |
| Stage 5B | More ops: Tile, CumSum, ConstantOfShape, SpaceToDepth, Pad, Slice, Resize, Sum |
| Stage 5C | CPU base class ops: Upsample, DepthToSpace |
| Stage 5D | Contrib ops: FastGelu, BiasDropout, SkipLayerNorm |

### 10.2 Running Tests

After building and deploying the plugin (see [Section 9.5](#95-deployment)):

```bash
# Run tests from /tmp to avoid module shadowing
cd /tmp
python /path/to/onnxruntime/test/python/transformers/test_cuda_plugin_ep.py
```

### 10.3 Parity Report

`tools/ci_build/cuda_plugin_parity_report.py` generates a report comparing registered kernels between the in-tree CUDA EP and the plugin EP, identifying gaps.

---

## 11. How to Add a New Kernel to the Plugin

### 11.1 If the kernel compiles as-is

Most kernels that don't use `GetComputeStream()` (returning `Stream*`) or inherit from excluded CPU base classes will compile without changes. The force-include mechanism handles type resolution automatically.

Just verify it's not in the exclusion list in `cmake/onnxruntime_providers_cuda_plugin.cmake`.

### 11.2 If the kernel calls a CPU base class helper

Apply the inline-header pattern:

1. Move the helper implementation from the CPU `.cc` file to the `.h` file
2. Wrap with `#ifdef SHARED_PROVIDER` (declaration only) / `#else` (inline body)
3. In the CUDA kernel, call the base class method directly (remove any local wrappers)
4. Verify all 4 build targets compile

Example from `cumsum.h`:
```cpp
namespace cumsum_op {
#ifdef SHARED_PROVIDER
Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out);
#else
inline Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) {
  // ... implementation ...
}
#endif
}
```

### 11.3 If the helper takes OpKernelContext&

Use a template version that accepts any context type:

```cpp
template <typename KernelContextType>
static void ComputePadsImpl(KernelContextType& ctx, ...) { ... }
```

The CUDA kernel calls `ComputePadsImpl(*ctx, ...)` directly.

### 11.4 If the kernel uses GetComputeStream()

Check whether the kernel actually dereferences the `Stream*` or just needs the raw `cudaStream_t`:

- If it only needs `stream->GetHandle()` → use `Stream(ctx)` instead (returns `cudaStream_t`)
- If it dereferences `CudaStream*` members → the kernel is blocked until the `Stream*` adapter is extended

### 11.5 If the kernel uses handle accessors

Use the plugin-compatible overloads already in `CudaKernel`:

```cpp
// Instead of:   GetCublasHandle(ctx->GetComputeStream())
// Use:          GetCublasHandle(ctx)  // or GetCublasHandle(Stream(ctx))
```

---

## 12. File Layout

```
onnxruntime/core/providers/cuda/plugin/
├── cuda_kernel_adapter.h        # CudaKernel base, macros, CPU shims (force-included)
├── cuda_ep_provider.h           # Plugin-local CUDAExecutionProvider
├── cuda_ep.h / .cc              # CudaEp : adapter::Ep
├── cuda_ep_factory.h / .cc      # CudaEpFactory : OrtEpFactory
├── cuda_plugin_ep.cc            # DLL entry points (CreateEpFactories/ReleaseEpFactory)
├── cuda_plugin_kernels.h / .cu  # Kernel registry creation
├── cuda_stream_plugin.h / .cc   # CudaSyncStream (handles, notifications)
├── cuda_allocator_plugin.h / .cc    # Device/pinned allocators
├── cuda_data_transfer_plugin.h / .cc # GPU↔CPU data transfer
├── cuda_controlflow_plugin.h / .cc / .cu  # If/Loop/Scan wrappers
├── cuda_graph_plugin.h / .cc    # CUDA Graph support
├── cuda_plugin_utils.h          # Common macros, error handling
├── cuda_iallocator_plugin.h     # IAllocator declarations
├── cuda_idata_transfer_plugin.h # IDataTransfer declarations
└── provider_api_shims.cc        # Reimplemented utility functions

include/onnxruntime/ep/
├── adapters.h                   # Master include + type aliasing (force-included)
├── api.h                        # ORT C API includes
├── common.h                     # EP common utilities
└── adapter/
    ├── allocator.h              # IAllocator adapter
    ├── data_transfer_manager.h  # DataTransferManager adapter
    ├── ep.h                     # Ep base class (wraps IExecutionProvider)
    ├── kernel_def.h             # KernelDef adapter
    ├── kernel_def_builder.h     # KernelDefBuilder adapter
    ├── kernel_registry.h        # KernelRegistry adapter
    ├── logging.h                # Logger adapter
    ├── node.h                   # Node adapter
    ├── op_kernel.h              # OpKernel + OpKernelContext adapters
    ├── op_kernel_info.h         # OpKernelInfo adapter
    └── tensor_helper.h          # Tensor creation from C API values
```

---

## 13. Future Work

1. **`Stream*` adapter** — Extend the adapter `OpKernelContext::GetComputeStream()` to return a full `Stream*` that attention/LLM ops can use. This unblocks ~25 operators.

2. **Tunable ops** — Implement a plugin-side `ITuningContext` and remove the `ORT_USE_EP_API_ADAPTERS` guards in `matmul.cc`/`gemm.cc`.

3. **String-array C API** — Add `KernelInfoGetAttributeArray_string` to the ORT C API to unblock RNN ops.

4. **Remaining CPU base classes** — Inline `SpaceDepthBase`, `UpsampleBase`, and object detection base classes.

5. **CI integration** — Add plugin build + test to the CI pipeline.

6. **CUDA Graph API for plugin EPs** — Add `IsGraphCaptureEnabled`, `IsGraphCaptured`, and `ReplayGraph` callbacks to the `OrtEp` C API (see [Section 5.4.4](#544-what-needs-to-change-in-ort-core-option-a)). This is required for efficient CUDA graph replay in the plugin EP. The capture/replay infrastructure (`cuda_graph_plugin.h/.cc`, `CudaEp` state machine) is already implemented and will activate once the API is extended.
