# CUDA Plugin EP — Design Document

## 1. Overview

The CUDA Plugin EP is an alternative build of the ONNX Runtime CUDA Execution Provider that compiles as a standalone shared library (`libonnxruntime_providers_cuda.so`). It loads at runtime through the ORT EP Plugin API instead of being statically linked into the main runtime binary.

**Goals:**
- Allow CUDA EP updates independent of ORT core releases
- Support all operators currently supported by the in-tree CUDA EP (tunable ops are low priority)
- Minimize changes to existing CUDA kernel source files

**Current status:** The plugin build is functional. Most core CUDA kernels now compile in the plugin build; the remaining source-level exclusions are documented in [Section 7](#7-excluded-operators).

---

## 2. Architecture

### 2.1 Build Targets

The ORT CUDA build produces four separate libraries:

| Target | Output | Type | Description |
|--------|--------|------|-------------|
| `onnxruntime_providers` | `libonnxruntime_providers.a` | Static lib | CPU provider + framework ops |
| `onnxruntime_providers_shared` | `libonnxruntime_providers_shared.so` | Shared lib | DLL-boundary bridge for in-tree EPs |
| `onnxruntime_providers_cuda` | `libonnxruntime_providers_cuda.so` | Shared module | In-tree CUDA EP (uses `SHARED_PROVIDER` bridge) |
| `onnxruntime_providers_cuda_plugin` | `libonnxruntime_providers_cuda.so` | Shared module | Plugin CUDA EP (uses EP API adapters) |

The plugin target keeps the canonical CUDA provider library filename (`onnxruntime_providers_cuda.*`) and advertises the canonical provider name (`CUDAExecutionProvider`). This is intentional compatibility behavior: a CUDA build contains either the plugin implementation or the legacy source-built implementation behind the same provider name and native filename, selected by `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN`.

### 2.1.1 Optional cuDNN Runtime Dependency

The CUDA Plugin EP follows the in-tree CUDA EP's optional-cuDNN model. cuDNN headers are still required at build time, but the plugin shared library must not link directly to cuDNN or contain a cuDNN DLL/SO in its dynamic dependency table. cuDNN is loaded lazily through the ORT cuDNN loader when `enable_cudnn` is enabled and the runtime libraries are available through trusted process-level library discovery.

The plugin exposes the same `enable_cudnn` provider option as the in-tree CUDA EP:

```text
enable_cudnn = 1  # default: try to load and use cuDNN when available
enable_cudnn = 0  # do not load cuDNN; run native CUDA paths or fail cuDNN-required ops with NOT_IMPLEMENTED
```

There is intentionally no provider option for a custom cuDNN DLL/SO path. Provider options can flow from higher-level configuration systems, so allowing them to choose a native library path would create a code-loading security risk. Deployments that need a specific cuDNN directory should use trusted process-level mechanisms, such as the OS loader configuration, container image setup, or Python `preload_dlls(cudnn=True, directory=...)` before plugin registration.

No-cuDNN plugin validation runs `test_cuda_plugin_ep.py` with `ORT_TEST_CUDA_PLUGIN_EP=1` and `ORT_TEST_CUDA_PLUGIN_NO_CUDNN=1`. That mode passes `enable_cudnn=0` to plugin sessions and skips tests for operators that still require cuDNN in the current implementation. Non-cuDNN operator coverage, plugin registration, device enumeration, graph assignment, CUDA graph, I/O binding, and profiling tests continue to run.

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
OrtEpFactory                     OrtEp
  ↑                                ↑
CudaEpFactory                    adapter::Ep
  │                                ↑
  ├─ creates OrtEpDevice           CudaEp
  ├─ provides fallback             ├─ stores session-derived Config
  │  CreateSyncStreamForDevice     ├─ implements OrtEp::CreateSyncStreamForDevice
  ├─ caches kernel registry        │  for per-session CudaSyncStream creation
  ├─ caches stable OrtMemoryInfo   ├─ synchronizes device (Sync)
  └─ maps OrtHardwareDevice*       └─ owns a real shim CUDAExecutionProvider via EpImpl()
       → CUDA ordinal

Migrated CUDA kernels
  └─ use CudaKernel / cuda_kernel_adapter.h
     ├─ cache a shared runtime-config handle during construction
     ├─ use CudaKernel accessors for provider settings during Compute()
     └─ resolve stream-local handles via CudaSyncStream::FromCudaStream()
```

Key ownership relationships:
- `CudaEpFactory` implements raw `OrtEpFactory` callbacks and owns shared factory-level state such as the kernel registry, cached `OrtMemoryInfo` instances, and the hardware-device to CUDA-ordinal map.
- `CudaEpFactory` also implements the factory-level `OrtEpFactory::CreateAllocator` and `OrtEpFactory::CreateSyncStreamForDevice` callbacks as fallback paths when the `OrtEp` callback is not used. Factory-created allocators are shared per device for internal arena/mempool use.
- `CudaEp` inherits from `ep::adapter::Ep`, which itself derives from `OrtEp` and owns a framework-facing `IExecutionProvider` object.
- `CudaEp` implements the `OrtEp::CreateSyncStreamForDevice` callback, which is the per-session stream-creation entry point used in preference to the factory callback.
- `CudaEp` implements `OrtEp::CreateAllocator` only when that EP instance was configured with `gpu_external_alloc` and `gpu_external_free`. External allocator callbacks are session-scoped provider options, so they are stored in `CudaEp::Config` and produce per-session `CudaExternalDeviceAllocator` instances rather than mutating shared factory device-cache state.
- The plugin-local `CUDAExecutionProvider` in `cuda_kernel_adapter.h` is a real shim object owned by `ep::adapter::Ep`. It is not the full in-tree CUDA EP, but it has its own object identity and stores plugin-specific members — including the wrapped `OrtEp*` and a provider-owned shared runtime-config object.
- Runtime configuration needed by migrated kernels is stored on that shim provider and exposed to kernels as a cached `shared_ptr<CudaKernelAdapterRuntimeConfig>`, rather than through a separate global map keyed by the provider address.
- `CudaSyncStream` owns `cudaStream_t`, `cublasHandle_t`, `cudnnHandle_t`, and `cublasLtHandle_t` for each sync stream created through the EP API.

### 2.4 Plugin DLL Entry Points

The plugin exports exactly two C symbols:
- `CreateEpFactories()` — called by ORT to create the EP factory
- `ReleaseEpFactory()` — called by ORT to destroy the factory

All other symbols have hidden visibility. In particular, the plugin does not export the legacy provider bridge entry point `GetProvider()`. Code running in ORT core or the Python binding must not assume that `GetProviderInfo_CUDA()` is available when the CUDA EP is supplied by this plugin library.

### 2.5 ORT Version Compatibility and API Negotiation

The plugin is built against the ORT headers in this repository (`ORT_API_VERSION`) but is designed to load into an **older** ORT runtime as well, down to the floor declared in [`plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION`](../../plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION) (currently `1.24.4`). The floor is a single source of truth:

- **Build time:** `cmake/onnxruntime_providers_cuda_plugin.cmake` reads the file and bakes it into the DLL as the `ORT_PLUGIN_EP_MIN_ORT_VERSION` preprocessor definition.
- **Packaging:** the Python wheel (`onnxruntime>=<floor>`, via `plugin-ep-cuda/python/build_wheel.py`) and the NuGet package (`plugin-ep-cuda/csharp/pack_nuget.py`) read the same file.

`CreateEpFactories()` negotiates the API version with the runtime instead of hard-coding it:

1. It calls `onnxruntime::ep::ApiInit(ort_api_base, ORT_PLUGIN_EP_MIN_ORT_VERSION)` (from `include/onnxruntime/ep/api.h`). `ApiInit()` parses the runtime version string reported by `OrtApiBase::GetVersionString()`, enforces the minimum, requests the `OrtApi` matching the **runtime's** version, and initializes the C++ API (`Ort::InitApi`). If the runtime is older than the floor, or the requested API/EP API is unavailable, `ApiInit()` throws and the factory creation fails with a descriptive `OrtStatus` (constructed conservatively via the v1 `OrtApi`, since the C++ API is not yet initialized at that point).
2. Every EP-facing callback struct (`OrtEpFactory`, `OrtEp`, `OrtAllocator`, `OrtSyncStreamImpl`, `OrtSyncNotificationImpl`, `OrtDataTransferImpl`, `OrtEpProfilerImpl`, `OrtLoopKernelHelper`, `OrtScanKernelHelper`) initializes its `ort_version_supported`/`version` field to `ORT_API_VERSION` — the ORT API version the plugin was compiled with — consistent with every other EP API struct. ORT uses this field only to avoid reading struct fields that did not exist when the plugin was compiled; it is append-only, so reporting the compiled version is always safe. Whether the plugin may *call* a newer `OrtApi`/`OrtEpApi` function is a separate concern governed by the runtime API version (`onnxruntime::ep::CurrentOrtApiVersion()`), handled by the capability gating in §2.6 — not by lowering this field.

> **Lowering the floor.** The floor reflects the newest `OrtApi`/`OrtEpApi` function the plugin actually calls. The kernel-registry EP path (`CreateKernelRegistry`, `KernelRegistry_AddKernel`, `GetKernelRegistry`, `EpGraphSupportInfo_*`, `CreateIfKernel`/`CreateLoopKernel`/`CreateScanKernel`) is `\since 1.24`; the stream, memory-device, and data-transfer EP functions are `\since 1.23`. The `Test Linux CUDA Plugin EP` stage (`plugin-linux-cuda-test-stage.yml`) installs the floor version of the base `onnxruntime` package and runs the plugin test against it, so an accidental dependency on a newer API is caught in CI. The Python plugin-loading helpers (`register_execution_provider_library`, `get_ep_devices`, `add_provider_for_devices`) must also exist in the floor's base package; the same test validates this.

### 2.6 API Version Audit and Defensive Capability Gating

Because the plugin binary may load into an older runtime, every `OrtApi`/`OrtEpApi` function it calls must exist in that runtime. The following audit records the newest `\since` version of each API surface the plugin uses (verified against the `\since` annotations in `onnxruntime_c_api.h` and `onnxruntime_ep_c_api.h`). It is the justification for the `1.24.4` floor and identifies exactly which features depend on APIs newer than the floor.

| API surface | Newest `\since` used | Representative functions |
| --- | --- | --- |
| `OrtApi` — direct calls (`ort_api_.*`, `Ort::GetApi().*`) | **1.23** | `SyncStream_GetHandle`, `GetTensorSizeInBytes`, `GetRunConfigEntry`, `CreateMemoryInfo_V2`, `Graph_GetNumNodes`/`Graph_GetNodes` (older: `CreateStatus`, `Logger_LogMessage`, `*KeyValuePairs`, `HardwareDevice_*`, `MemoryInfoGet*`, `GetSessionConfigEntry`) |
| `OrtApi` — optional gated kernel-context capability | **1.28** | `KernelContext_GetSyncStream` (called from the adapter only when `CurrentOrtApiVersion() >= 28`; otherwise scratch allocation uses a null stream tag and concurrent run support is not advertised) |
| `OrtEpApi` — direct calls (`ep_api_.*`, `Ort::GetEpApi().*`) | **1.24** | `CreateKernelRegistry`, `KernelRegistry_AddKernel`, `ReleaseKernelRegistry`, `CreateIfKernel`/`CreateLoopKernel`/`CreateScanKernel`, `EpGraphSupportInfo_LookUpKernel` (older: `MemoryDevice_*`, `MemoryInfo_GetMemoryDevice`, `SyncStream_*`, `EpDevice_AddAllocatorInfo`, `EpGraphSupportInfo_AddSingleNode`, `CreateEpDevice`/`ReleaseEpDevice`) |
| EP profiler API (only when built with `ENABLE_CUDA_PROFILING`) | **1.25** | `CreateProfilingEvent`, `ProfilingEventsContainer_AddEvents`, `ReleaseProfilingEvent` (called from `cuda_profiler_plugin.cc` via the `Ort::ProfilingEvent` / `Ort::UnownedProfilingEventsContainer` wrappers) |

`provider_api_shims.cc` uses only internal helpers (`GetEnvironmentVar`, `MLFloat16` conversions), and the plugin uses no Model Editor, Model Package, or Compile API. **Apart from optional gated capabilities such as EP profiling and stream-tagged scratch allocation, every API the plugin calls is `\since 1.24` or older**, so the true compatibility floor is `1.24.4`.

**Defensive capability gating.** Reading a struct field is safe because the field is append-only and ORT only reads fields it knows about. The real hazard is *calling* an `OrtApi`/`OrtEpApi` function that the (possibly older) runtime does not provide. The correct guard for that is the runtime API version, `onnxruntime::ep::CurrentOrtApiVersion()`, not `ort_version_supported`. The `CudaEp` constructor (`cuda_ep.cc`) therefore reads `const uint32_t ort_version = onnxruntime::ep::CurrentOrtApiVersion();` and only installs an `OrtEp` callback when that runtime version is new enough to provide both the callback field and every API its implementation calls:

| `OrtEp` callback | `\since` | Installed only when |
| --- | --- | --- |
| `Sync` | 1.25 | `ort_version >= 25` |
| `CreateProfiler` | 1.25 | `ort_version >= 25` **and** `ENABLE_CUDA_PROFILING` |
| `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy` | 1.26 | `ort_version >= 26` |
| `GetAvailableResource` | 1.26 | `ort_version >= 26` |

All other `OrtEp` and `OrtEpFactory` callbacks are `\since 1.24` or older and are installed unconditionally. Gating `CreateProfiler` is what makes the three `\since 1.25` profiler functions unreachable on an older runtime: when the profiler is never created, ORT never drives the `OrtEpProfilerImpl` callbacks that call them.

`KernelContext_GetSyncStream` is guarded at the adapter call site rather than through an `OrtEp` callback field: `OpKernelContext::GetSyncStream()` returns null when `CurrentOrtApiVersion() < 28`, and `CudaEp::IsConcurrentRunSupportedImpl()` only advertises concurrent runs when that API is available. Older runtimes therefore keep the previous serialized-run behavior while still using the same plugin binary.

The gates use **graceful degradation rather than throwing**: the gated callbacks and adapter capabilities are optional features (per-run sync, EP-level GPU profiling, CUDA-graph capture/replay, device-memory budgeting, stream-tagged scratch for concurrent runs), so disabling them on an older runtime still yields a fully functional EP — inference runs, just without that specific feature. This was validated by loading the plugin (built against the latest headers) into both the latest runtime (full test suite passes) and an `onnxruntime==1.24.4` runtime (the EP registers, enumerates devices, and runs inference correctly with the newer callbacks left null).

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

This 1100+ line header provides everything CUDA kernels need that would normally come from framework infrastructure:

| Section | What It Provides |
|---------|-----------------|
| Error macros | `CUDA_RETURN_IF_ERROR`, `CUBLAS_RETURN_IF_ERROR`, `CUDNN_RETURN_IF_ERROR`, `CUFFT_RETURN_IF_ERROR` |
| Type mappings | `ToCudaType<MLFloat16>::MappedType = half`, etc. |
| CudaKernel base | Scratch buffers, handle access, `Stream()`, `GetComputeStream()` |
| Kernel registration | Self-registering `ONNX_OPERATOR_*_KERNEL_EX` macro overrides via `PluginKernelCollector`, including `ONNX_OPERATOR_TWO_TYPED_KERNEL_EX`, `ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX`, and `ONNX_OPERATOR_THREE_TYPED_KERNEL_EX` variants |
| CPU shims | Lightweight reimplementations of CPU helpers not linked into plugin |
| Math helpers | `HalfGemmOptions`, `CublasMathModeSetter` |
| Stream shim | `OrtStreamAdapter`/`PluginStreamShim` to present a framework-compatible `Stream*` view over a raw `cudaStream_t` where needed |

### 3.5 Kernel Registration

In the in-tree build, kernels register through centralized tables (`cuda_nhwc_kernels.cc`, `cuda_contrib_kernels.cc`). In the plugin build, the `ONNX_OPERATOR_*_KERNEL_EX` macros are overridden to auto-register each kernel into the `PluginKernelCollector` singleton at static initialization time:

```cpp
// Macro override generates:
// 1. BuildKernelCreateInfo<CLASS>() function
// 2. Static PluginKernelCollector::Register() call

// At plugin startup, CreateCudaKernelRegistry() iterates the collector
// and registers each kernel into an adapter::KernelRegistry.
```

#### 3.5.1 Type Constraint Names and OpSchema Access

Every kernel registration includes type constraint names — string literals such as `"T"`, `"T1"`, `"U"` — that must exactly match the formal parameter type-constraint strings defined in the ONNX operator schema. In the current plugin build, these names are **hard-coded** at compile time with no runtime validation against the actual schema. If a constraint name is wrong, kernel matching silently fails during `GetCapability`.

PR #27713 adds `OrtEpApi` functions that let plugin EPs query ONNX operator schemas from ORT's global schema registry at runtime (available from ORT 1.25):

| `OrtEpApi` Function | C++ Wrapper | Purpose |
|---------------------|-------------|----------|
| `GetOpSchema(name, max_ver, domain)` | `Ort::GetOpSchema()` | Look up a schema by op name, max opset version, and domain |
| `OpSchema_GetSinceVersion` | `ConstOpSchema::GetSinceVersion()` | Opset version that introduced this schema entry |
| `OpSchema_GetNumInputs` / `_GetNumOutputs` | `GetNumInputs()` / `GetNumOutputs()` | Formal input/output count |
| `OpSchema_GetInputName` / `_GetOutputName` | `GetInputName(i)` / `GetOutputName(i)` | Formal parameter name |
| `OpSchema_GetInputTypeStr` / `_GetOutputTypeStr` | `GetInputTypeStr(i)` / `GetOutputTypeStr(i)` | Type constraint string (e.g., `"T"`) |
| `OpSchema_HasTypeConstraint` | `HasTypeConstraint(str)` | Whether a string is a valid type constraint name in the schema |

The returned `OrtOpSchema*` is non-owning — it points into the global `ONNX_NAMESPACE::OpSchemaRegistry` singleton and is valid for the lifetime of the ORT process.

**Why the plugin cannot link its own ONNX library:** The `OpSchemaRegistry` is a Meyers singleton (`static` local in `Instance()`). Each shared library gets its own copy of that static variable — on Windows each DLL is isolated by default, on macOS two-level namespaces have the same effect, and on Linux behavior depends on `dlopen` flags. Even when isolation doesn't occur, the EP's registry would lack ORT's contrib and internal schemas, and version mismatches between the EP's ONNX library and ORT's vendored copy could cause silent divergence. The `OrtEpApi` route is the only reliable, portable way to query the schemas ORT actually uses.

**Impact on the CUDA plugin EP:**

1. **Registration-time validation.** `CreateCudaKernelRegistry()` can optionally validate each registered kernel's type constraint names against the schema after collecting all entries from `PluginKernelCollector`. A mismatch can be logged as a warning (debug builds) or an error, catching drift when ONNX spec updates rename constraint strings.

2. **NHWC / internal-domain diagnostics.** For rewritten `com.ms.internal.nhwc` nodes, the schema API can confirm that the kernel's registered domain, version range, and constraint names actually match the internal-domain schema entry, improving the diagnostics called for in [Section 5.3.1.3](#5313-nhwc-design-requirements).

3. **Parity tooling.** `cuda_plugin_parity_report.py` can use the C++ wrapper to compare the plugin's registered constraint names against the schema, flagging incorrect or missing constraints in the parity report.

4. **Future: schema-driven registration helpers.** A `KernelDefBuilder` helper could derive constraint names automatically from the schema rather than relying on hard-coded strings, reducing the manual maintenance burden when new opset versions change constraint names. See [Section 11.6](#116-if-opschema-access-is-available-schema-validated-type-constraints).

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
- `space_depth_ops.h` — `SpaceDepthBase` constructor plus shared `ReadBlocksize`, `ReadIsDCR`, and dimension-validation helpers (templatized on info/context type where needed)
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

- **`ConstantOfShapeBase`** — depends on `TensorProto` and `UnpackTensor`. The plugin path in `constant_of_shape.h` stays self-contained: it reuses `ConstantOfShapeCore` but fetches the `value` attribute through the ORT C++ API instead of depending on the full CPU base implementation.

`UpsampleBase` no longer belongs in this category: the adapter now exposes `OpKernelInfo::GetAllocator(OrtMemType)`, and the remaining shape-rank query already has an adapter-safe fallback when `Node::InputDefs()` is unavailable. That lets the CUDA `Upsample` antialias path reuse the same persistent device lookup-table initialization in both bundled and plugin builds instead of keeping a plugin-only scratch-buffer fallback.

---

## 5. Handle and Stream Management

### 5.1 Stream Ownership

`CudaSyncStream` is the plugin's CUDA sync-stream implementation:
- Owns `cudaStream_t`, `cublasHandle_t`, `cudnnHandle_t`, `cublasLtHandle_t`
- Can be created at two levels:
  - **OrtEp-level** via `CudaEp::CreateSyncStreamForDeviceImpl` for per-session stream creation
  - **OrtEpFactory-level** via `CudaEpFactory::CreateSyncStreamForDeviceImpl` as the fallback path
- Registers itself in a global `cudaStream_t -> CudaSyncStream*` map so migrated kernels can recover per-stream handles from a raw CUDA stream
- Defers host-buffer cleanup until `OnSessionRunEnd()` after the stream is synchronized

#### 5.1.1 Device Synchronization (Sync API)

`CudaEp` implements `OrtEp::Sync` to block until the configured CUDA device has completed all preceding work. ORT uses this path for scenarios such as `IOBinding`, where asynchronous input copies must finish before kernel execution begins.

Implementation details:
- `CudaEp::SyncImpl` temporarily switches to the EP's configured device via `cudaSetDevice(device_id)` and restores the caller's previous CUDA device before returning
- It then issues `cudaDeviceSynchronize()` as a conservative device-wide barrier

This is intentionally conservative and correct for the plugin EP's first sync integration. A narrower stream-scoped synchronization strategy can be considered later if profiling shows a need.

#### 5.1.2 Python Host/Device Tensor Copies

The Python `OrtValue` helpers (`update_inplace()` for host-to-device and `numpy()` for device-to-host) historically reached CUDA copies through the legacy provider bridge (`GetProviderInfo_CUDA()`). That bridge requires the provider shared library to export `GetProvider()`, which the CUDA plugin intentionally does not export.

To keep working when the bridge is absent (as with the plugin EP), the pybind can reach CUDA copies two ways: the legacy provider bridge (`TryGetProviderInfo_CUDA()`) and a plugin-registered `OrtDataTransfer` copy function (`CreateDataTransferMemCpy()`, backed by the plugin EP's `IDataTransfer`). It tries whichever is available and throws if neither is, in which case a CUDA `OrtValue` copy cannot be performed.

The two code paths differ only in which mechanism they try first, and this does not change the outcome (exactly one applies in a given build):

- `OrtValue.update_inplace(numpy_array)` / `OrtValue.numpy()` (in `onnxruntime_pybind_ortvalue.cc`) try the provider bridge first, then fall back to the plugin `OrtDataTransfer`.
- `OrtValue.update_inplace(OrtValue)` (`UpdateOrtValueInplace` in `onnxruntime_pybind_mlvalue.cc`) tries the plugin `OrtDataTransfer` first, then falls back to the built-in CUDA provider copy functions.

### 5.2 Handle Access Path

```
CudaKernel::GetCublasHandle(OpKernelContext* ctx)
  → Stream(ctx)                           // raw cudaStream_t from adapter ctx
  → CudaSyncStream::FromCudaStream()      // global stream map + TLS cache
  → sync_stream->GetCublasHandle()
```

The stream lookup path uses a thread-local last-hit cache plus a generation counter so destroyed streams invalidate stale TLS entries without requiring per-thread cleanup.

For code paths that need handles without an active stream, `cuda_kernel_adapter.h` also provides thread-local default cuBLAS/cuDNN handles via `GetDefaultCudaHandlesForDevice(device_id)`.

### 5.3 Provider Access

Kernels still discover the shim provider through the pointer returned by `info.GetExecutionProvider()` at construction time. In the plugin build, `ep::adapter::OpKernelInfo` snapshots three related pointers from the framework `OpKernelInfo` when the kernel is created:

- the session-owned outer `PluginExecutionProvider`
- the wrapped plugin `OrtEp` / `CudaEp`
- the inner shim provider returned by `static_cast<const Ep*>(ort_ep)->EpImpl()`

`OpKernelInfo::GetExecutionProvider()` then returns that cached shim pointer, so migrated kernels receive the real shim `CUDAExecutionProvider` object owned by `ep::adapter::Ep`, not the outer `PluginExecutionProvider` and not the `OrtEp`/`CudaEp` object reinterpreted at the same address.

Caching the shim pointer at kernel-creation time is important for the NHWC path. Re-querying `OrtKernelInfo -> OrtEp -> EpImpl()` during execution was fragile after layout transformation. The current implementation resolves the shim once in the `CudaKernel` constructor, caches a `shared_ptr<CudaKernelAdapterRuntimeConfig>`, and routes later provider-setting reads through `CudaKernel` accessors instead of repeated provider-pointer casts during `Compute()`.

This changes the safety model from the earlier "phantom shim" design:
- The shim no longer needs to remain layout-compatible with `IExecutionProvider`.
- Adding plugin-local members to the shim is safe as long as normal C++ object lifetime/ownership rules are respected.
- The shim still is not the full bundled `CUDAExecutionProvider`; it only exposes the subset of methods that migrated kernels currently need.

Provider options flow through the plugin in two stages:
- `CudaEpFactory` parses session/provider options into `CudaEp::Config`.
- `CudaEp` copies the subset needed by migrated kernels into the shim provider's runtime config via `SetCudaKernelAdapterRuntimeConfigForProvider(EpImpl(), ...)` during EP construction.

Because the runtime config is provider-owned and cached by kernels as a shared pointer, there is no global map and no mutex. Today that stored subset includes TF32, device ID/device properties, cuDNN convolution settings, skip-layer-norm strict mode, fused-conv-bias, and SDPA kernel selection. Other plugin behaviors, such as preferred layout, are handled directly by `CudaEp` callbacks instead of through the shim.

For stream bridging, the preferred helpers are:
- `Stream(ctx)` when the kernel only needs a raw `cudaStream_t`
- `GetComputeStream(ctx)` when the kernel API already accepts the adapter's opaque stream pointer
- `GetOrtStream(ctx)` when framework-style `Stream*` plumbing is still needed by shared helper code

#### 5.3.1 NHWC Layout-Transformation Support

The bundled CUDA EP's NHWC path is not just a kernel-registration feature. It is a coordinated contract between provider configuration, ORT's layout transformer, kernel registration, adapter/provider access, and graph partitioning. The current implementation supports this path when NHWC is compiled in and the session requests `prefer_nhwc`.

#### 5.3.1.1 End-to-End Flow

When NHWC is enabled for an EP, the expected ORT flow is:

1. The EP reports `NHWC` from `GetPreferredLayout()` (or `OrtEp::GetPreferredDataLayout()` for plugins) when `prefer_nhwc` is enabled.
2. During layout transformation, ORT asks the EP whether each layout-sensitive op should be converted via `ShouldConvertDataLayoutForOp()`.
3. For each accepted op, `TransformLayoutForEP()` inserts `Transpose` nodes around the operator and rewrites the operator into the internal NHWC domain (`com.ms.internal.nhwc`).
4. Graph partitioning runs again. The EP must now claim the rewritten internal-domain nodes, not the original ONNX-domain nodes.
5. Kernel lookup must succeed against the EP's kernel registry for the rewritten internal-domain node, with matching domain, opset range, and type constraints.

This means the plugin must satisfy two distinct contracts at the same time:

| Contract | Owner | Requirement |
|----------|-------|-------------|
| Layout preference contract | `CudaEp` + ORT plugin bridge | Only request NHWC when the plugin can handle the rewritten graph |
| Kernel/capability contract | Kernel registry + `CudaEp::GetCapabilityImpl()` | Claim the resulting `com.ms.internal.nhwc` nodes during partitioning |

#### 5.3.1.2 Current Plugin Status

The current implementation already has the core runtime pieces in place:

| Component | Current state |
|-----------|---------------|
| ORT plugin bridge | `PluginExecutionProvider` already maps `OrtEp::GetPreferredDataLayout()` and `OrtEp::ShouldConvertDataLayoutForOp()` into the normal `IExecutionProvider` layout APIs |
| Plugin callback implementations | `CudaEp` installs `GetPreferredDataLayoutImpl()` and `ShouldConvertDataLayoutForOpImpl()` and advertises NHWC when `prefer_nhwc` is enabled |
| Provider option parsing | `CudaEpFactory` already parses `prefer_nhwc` / `prefer_nhwc_layout` into `CudaEp::Config` |
| Build-time gating | `cmake/onnxruntime_providers_cuda_plugin.cmake` propagates `ENABLE_CUDA_NHWC_OPS` to the plugin target when `onnxruntime_USE_CUDA_NHWC_OPS=ON` |
| NHWC kernel registration | NHWC kernels are compiled from the normal CUDA kernel sources and self-register through `PluginKernelCollector`; the centralized `cuda_nhwc_kernels.cc` table stays excluded in plugin builds |
| Second capability pass | `CudaEp::GetCapabilityImpl()` preserves nodes already assigned to `CUDAExecutionProvider`, so ORT's post-layout-transformation partitioning pass does not drop rewritten NHWC nodes that were previously selected by the plugin |
| Adapter provider access | `ep::adapter::OpKernelInfo` caches the inner shim `EpImpl()` pointer at kernel-creation time, avoiding a fragile runtime `OrtKernelInfo -> OrtEp -> EpImpl()` round-trip in NHWC kernels |
| Focused validation | `test_cuda_plugin_ep.py` Stage 3 now runs NHWC-requested sessions for Conv, BatchNormalization, MaxPool, and AveragePool and requires plugin-backed execution to succeed numerically |

The fixes that made this work were not limited to turning the callbacks back on:

- The plugin now keeps both newly discovered candidate nodes and nodes already assigned to `CUDAExecutionProvider` during the second `GetCapability()` pass that runs after layout transformation.
- NHWC kernels now obtain provider configuration through the cached shim pointer in `ep::adapter::OpKernelInfo`, which removed a runtime crash path in migrated kernels such as NHWC `Conv`.

With those pieces in place, NHWC-requested sessions take the real plugin execution path rather than silently falling back to the stable NCHW path.

#### 5.3.1.3 NHWC Design Requirements

The implementation should preserve the following invariants:

| Requirement | Why it matters |
|-------------|----------------|
| The plugin must never advertise NHWC unless it can claim every internal-domain op it requests ORT to generate | Otherwise ORT can create an invalid graph containing `com.ms.internal.nhwc` nodes that no EP selects |
| The NHWC conversion allowlist must come from a single shared source of truth | The bundled CUDA EP and the plugin EP must not drift on which ops are safe to rewrite |
| Kernel coverage checks must validate internal-domain registrations, not just original ONNX-domain registrations | The rewritten graph uses `com.ms.internal.nhwc`, so ONNX-domain coverage alone is insufficient |
| Capability diagnostics must identify internal-domain kernel misses clearly | NHWC failures are difficult to debug after rewrite unless the missing domain/op/version/type information is surfaced |
| Tests must verify plugin-backed NHWC execution explicitly | Output correctness alone is not enough because a fallback path can still pass numerically |

#### 5.3.1.4 Implemented Design and Remaining Follow-Ups

The current implementation has the minimum runtime fixes required for plugin-side NHWC execution. The remaining work is mostly cleanup, consolidation, and stronger diagnostics.

**A. Keep partitioning registry-driven and preserve pre-assigned NHWC nodes**

`CudaEp::GetCapabilityImpl()` should continue to rely on `EpGraphSupportInfo_LookUpKernel()` as the source of truth for whether a rewritten node is supported. The important implementation detail is that it must preserve nodes already assigned to the plugin when ORT reruns partitioning after layout transformation.

That behavior is now implemented by tracking:
- `tentative_nodes`: newly discovered nodes with matching kernel registrations
- `candidate_nodes`: both tentative nodes and nodes already assigned to `CUDAExecutionProvider`

The final support set is chosen from `candidate_nodes`, with the existing CPU-preferred-node filtering applied only where appropriate.

When resource accounting is also enabled (`session.resource_cuda_partitioning_settings`), this two-pass flow interacts with the partitioner's budget commit in an important way. The first-pass tags are tentative — ORT applies them only so the layout transformer can rewrite the nodes — so the partitioner does **not** commit any accountant budget for them. After the second pass, the partitioner commits budget only for the first-pass nodes that survived (still claimed by the plugin), using the per-node costs captured during the first pass. Nodes dropped on the second pass therefore never consume budget, and surviving nodes are counted exactly once. Plugin EPs that attach accounting costs should do so only on first-pass (newly claimed) capabilities, mirroring the in-tree CUDA EP, which leaves already-assigned second-pass nodes cost-free and relies on the partitioner's deferred commit.

**B. Cache the shim provider pointer at kernel creation**

Migrated CUDA kernels expect `info.GetExecutionProvider()` to return the shim `CUDAExecutionProvider`, not the outer `PluginExecutionProvider`. The adapter now resolves that relationship once during kernel creation, captures the shim provider's runtime-config object, and uses `CudaKernel` accessors for later provider-setting reads.

This is especially important for NHWC kernels because layout transformation introduces additional runtime paths before the actual CUDA kernel executes. Repeatedly reconstructing provider access from `OrtKernelInfo` during execution proved fragile in that path. The cached-config approach keeps provider access deterministic and matches the actual object model:

- outer session EP: `PluginExecutionProvider`
- wrapped plugin object: `CudaEp` / `ep::adapter::Ep`
- inner shim: `CUDAExecutionProvider` returned by `EpImpl()`

**C. Remaining follow-ups**

The main follow-ups are now design quality items rather than blockers:

- Unify the NHWC conversion allowlist between the bundled CUDA EP and the plugin CUDA EP instead of keeping separate hard-coded tables.
- Improve diagnostics when kernel lookup fails for a rewritten `com.ms.internal.nhwc` node.
- Extend tests to assert internal-domain rewrite structure directly, not just plugin-backed execution and numerical correctness.

#### 5.3.1.5 Rollout Status

The NHWC rollout is effectively in a "runtime enabled, cleanup remaining" state:

| Phase | Change | Expected outcome |
|-------|--------|------------------|
| 1 | Enable plugin NHWC callbacks and preserve pre-assigned nodes in the second capability pass | Implemented |
| 2 | Cache the shim provider pointer in the adapter `OpKernelInfo` | Implemented; fixes the observed NHWC runtime crash |
| 3 | Consolidate allowlists, improve internal-domain diagnostics, and strengthen structural NHWC assertions | Recommended follow-up work |

#### 5.3.2 Allocator Resolution for Kernels (Scratch and PrePack)

Migrated kernels need a valid device allocator in two places: scratch/workspace buffers during `Compute()`, and one-time weight conversion or packing during `PrePack()`. Both now resolve the allocator the same way the bundled CUDA EP does, through the kernel's own `OpKernelInfo`.

- **Scratch buffers.** `CudaKernel::GetScratchBuffer` allocates through `Info().GetAllocator(OrtMemTypeDefault)` (the EP arena) and stream-tags scratch chunks with the framework `OrtSyncStream*` from `KernelContext_GetSyncStream`, instead of issuing a raw `cudaMallocAsync`/`cudaMalloc` per call. The adapter `OpKernelInfo::GetAllocator` resolves the EP's default-memory (device) allocator and is always valid for a migrated kernel, so no plugin-only scratch path is needed. Routing through the arena is also what keeps the device free-memory footprint stable during CUDA graph capture (see [cuda_graph_for_cuda_plugin.md](cuda_graph_for_cuda_plugin.md#arena-allocator-integration)). CUDA launches still use the raw `cudaStream_t` from `KernelContext_GetGPUComputeStream`; the framework stream is used only for stream-aware arena bookkeeping.
- **PrePack.** The framework prepack loop (`SessionState::PrepackConstantInitializedTensors`) resolves the allocator with `GetInitializerAllocator(kernel->Info().GetDevice(OrtMemTypeDefault))`, a session map keyed by device. For a plugin EP registered as a separate library, that device-keyed lookup can miss and return null. The loop now falls back to `kernel->Info().GetAllocator(OrtMemTypeDefault)` when the lookup is null, so every `PrePack` implementation receives a valid allocator at the single framework call site. This replaces the earlier approach of adding a per-kernel `if (!alloc) alloc = Info().GetAllocator(...)` guard to each prepacking op (which only covered the few ops that were touched and risked missing future ones). The fallback is behavior-neutral for in-tree EPs, whose device-keyed lookup already succeeds, and it does **not** force `is_packed`/`prepacked_weights` handling \u2014 ops such as `QMoE` and `MatMulNBits` still set `is_packed = true` and populate prepacked weights normally.

The enabling adapter changes are in [`include/onnxruntime/ep/adapter/allocator.h`](../../include/onnxruntime/ep/adapter/allocator.h) and [`include/onnxruntime/ep/adapter/op_kernel.h`](../../include/onnxruntime/ep/adapter/op_kernel.h): `IAllocatorWrappingOrtAllocator` implements `IsStreamAware()`/`AllocOnStream()` by forwarding to the underlying `OrtAllocator`'s `AllocOnStream` when it is available (ORT >= 1.23), and `OpKernelContext::GetSyncStream()` exposes the framework stream when the negotiated ORT API version includes `KernelContext_GetSyncStream`. The CUDA plugin uses that framework stream for `GetScratchBuffer`; if it is unavailable, allocation falls back to a null stream tag and concurrent `Session::Run()` is not advertised.

### 5.4 CUDA Graph Support

CUDA Graph capture/replay is fully implemented for the plugin EP, including arena integration (both default BFC arena and CUDA native mempool), multi-graph via annotation IDs with different input shapes, and combining a caller-supplied `user_compute_stream` with capture/replay. Concurrent `Session::Run()` is supported when the host runtime exposes `KernelContext_GetSyncStream` and the session is not forced into EP-level unified-stream mode. The full design — plugin-side implementation, per-thread isolation, arena integration, capture flow, and user-stream mode — is in [cuda_graph_for_cuda_plugin.md](cuda_graph_for_cuda_plugin.md). This section documents only the framework-level and C API changes that affect the broader ORT architecture.

#### 5.4.1 OrtEp C API Extensions (v1.26)

Four new optional callbacks in `OrtEp` (`onnxruntime_ep_c_api.h`):

| Callback | Signature | Default (NULL) | Purpose |
|----------|-----------|----------------|---------|
| `IsGraphCaptureEnabled` | `bool(const OrtEp*)` | `false` | Report whether graph capture is enabled |
| `IsGraphCaptured` | `bool(const OrtEp*, int graph_annotation_id)` | `false` | Check if a graph has been captured for a given annotation ID |
| `ReplayGraph` | `OrtStatus*(OrtEp*, int graph_annotation_id)` | OK | Launch a previously captured graph |
| `GetGraphCaptureNodeAssignmentPolicy` | `OrtGraphCaptureNodeAssignmentPolicy(const OrtEp*)` | `ALL_NODES_ON_EP` | Specify validation strictness for node assignment |

These are supplemented by the existing `OnRunStart` / `OnRunEnd` lifecycle callbacks that drive the capture workflow.

The `PluginExecutionProvider` bridge (`ep_plugin_provider_interfaces.cc`) delegates to these callbacks with version gating (`ort_version_supported >= 26`), falling back to safe defaults for older plugins.

#### 5.4.2 Framework Changes

The `IExecutionProvider` base class gained a `GetGraphCaptureNodeAssignmentPolicy()` virtual (default: `ALL_NODES_ON_EP`). All in-tree EPs with graph capture (CUDA, DML, JS, WebGPU) override to `ALLOW_CPU_FOR_SHAPES`.

Session-level changes in `inference_session.cc`:

- **Policy-driven validation**: Graph capture validation at session initialization now iterates all EPs and queries `GetGraphCaptureNodeAssignmentPolicy()` instead of hard-coding EP name lists.
- **Bounded recursion**: After each normal run when graph capture is enabled, the session recursively calls `RunImpl()` (bounded by `kMaxGraphCaptureWarmupRuns = 8`) until the graph is captured. From the user's perspective, a single `Run()` call handles the entire warm-up + capture sequence.
- **Stream collection lifetime**: ORT core now caches `DeviceStreamCollection` objects in thread-affine session buckets keyed by a per-thread lifetime token. Graph-enabled runs recycle and reacquire stream wrappers only on the creating thread, which preserves warm-up/capture reuse without cross-thread leakage.

#### 5.4.3 User Compute Stream with CUDA Graph

A caller-provided `user_compute_stream` may be combined with `enable_cuda_graph` (the factory previously rejected this pair). When both are set, `CudaEp::GetPerThreadContext()` builds the per-thread graph context around the user-owned stream rather than an EP-owned one, so capture and replay run on the same stream the kernels are issued to (matching the bundled CUDA EP). The context marks the stream as not owned and never destroys it. Details are in [cuda_graph_for_cuda_plugin.md](cuda_graph_for_cuda_plugin.md#user-compute-stream--cuda-graph).

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
| `ConstOpSchema` | `const OrtOpSchema*` | `GetSinceVersion()`, `GetNumInputs()`, `GetInputName()`, `GetInputTypeStr()`, `HasTypeConstraint()` (ORT ≥ 1.25, PR #27713) |

---

## 7. Excluded Operators

Section 7 reflects the current source exclusions in `cmake/onnxruntime_providers_cuda_plugin.cmake`, plus the small set of intentionally out-of-scope directories. This is the source of truth for what the plugin build omits today.

### 7.1 Infrastructure (Permanently Excluded — Replaced by Plugin Equivalents)

| File | Reason |
|------|--------|
| `cuda_execution_provider.cc` | Replaced by `cuda_ep.h/.cc` and the plugin adapter/runtime shim |
| `cuda_provider_factory.cc` | Replaced by `cuda_ep_factory.cc` |
| `cuda_provider_interface.cc` | Not needed in plugin architecture |
| `cuda_stream_handle.cc` | Replaced by `cuda_stream_plugin.cc` |
| `cuda_execution_provider_info.cc` | Config parsed directly in `CudaEp::Config` |
| `cuda_graph.cc` | Replaced by plugin-local `cuda_graph_plugin.h/.cc`, which implements graph capture/replay through the OrtEp graph callbacks |
| `cuda_mempool_arena.cc` | Replaced by plugin-native `cuda_mempool_allocator_plugin.h/.cc` (uses CUDA mempool directly behind `OrtAllocator`) |
| `cuda_common.cc` | Utility functions shimmed in `cuda_kernel_adapter.h` |
| `cuda_nhwc_kernels.cc` | Replaced by `PluginKernelCollector` auto-registration |
| `cuda_contrib_kernels.cc` | Replaced by `PluginKernelCollector` auto-registration |

### 7.2 Pure CPU Ops (Permanently Excluded)

| File | Reason |
|------|--------|
| `tensor/size.cc` | Pure CPU op, handled by `GetCpuPreferredNodes` |
| `tensor/shape_op.cc` | Pure CPU op, inherits from `onnxruntime::OpKernel` (framework) |

### 7.3 Additional Current Source Exclusions

| File / Pattern | Why It Is Excluded Today | What Would Unblock It |
|----------------|--------------------------|------------------------|
| `core/providers/cuda/controlflow/*` | The framework controlflow kernels inherit from CPU-side controlflow bases (`If`, `Loop`, `Scan`) and are intentionally omitted from the plugin source list | No change is currently planned. The plugin uses its own `cuda_controlflow_plugin.cc` wrappers instead of these framework sources |
| `tunable/*` | Depends on the real `CudaTuningContext` and other framework CUDA EP infrastructure that is not available in the plugin build | Add a plugin-capable tuning context and remove the remaining framework-only tunable dependencies |
| `tensor/sequence_op.cc` | Uses `TensorSeq`, which is still not adapter-safe here | Add `TensorSeq` adapter coverage |
| `contrib_ops/cuda/llm/*` | Contrib LLM sources have not gone through the same adapter-migration pass as the core CUDA LLM kernels | Finish contrib-LLM-specific adapter work |
| `contrib_ops/cuda/tensor/shrunken_gather.cc` | The training-side header path still depends on framework/provider API wiring | Low-priority training-specific adapter work |

| `contrib_ops/cuda/transformers/*` | Beam search, greedy search, and sampling depend on broader framework/subgraph integration that has not been adapted for the plugin build | Significant adapter and subgraph support work |
| `contrib_ops/cuda/aten_ops/*` | ATen interop is intentionally out of scope for the standalone CUDA plugin build | A separate ATen/plugin strategy |
| `contrib_ops/cuda/collective/*` | Collective/NCCL support is intentionally out of scope for the standalone CUDA plugin build | A separate collective/NCCL plugin strategy |

### 7.4 Common Exclusion Themes

The current exclusions fall into a few categories:

1. **Tunable/framework-dependent infrastructure** — `tunable/*`, contrib transformers, and some contrib LLM paths still rely on framework-only execution-provider services.

2. **Remaining adapter gaps** — `TensorSeq` (needed for `sequence_op.cc`) and contrib-LLM-specific plumbing still need dedicated adapter work.

3. **Deliberate scope cuts** — ATen and collective/NCCL sources remain intentionally out of scope for the standalone CUDA plugin.

---

## 8. Remaining `#ifdef` Guards in Kernel Code

The branch still contains a small set of plugin guards in both infrastructure and operator code. The important pattern has not changed:

- Infrastructure files such as `cuda_kernel.h`, `cuda_common.h`, and `cudnn_common.h` still need build-mode gates.
- `generator/constant_of_shape.h` still needs a plugin-specific path because `ConstantOfShapeBase` depends on framework-only tensor-attribute helpers.
- Tunable kernels such as `math/matmul.cc` still gate framework-only registration paths.
- `tensor/identity_op.h` guards the `TensorSeq` code path and `context->InputType()` call with `#ifndef BUILD_CUDA_EP_AS_PLUGIN` — the plugin build handles only the `Tensor` path. `identity_op.cc` uses conditional macros (`IDENTITY_V_TYPES` / `IDENTITY_V_TYPES_IRv9`) so opset 14+ registrations use `AllFixedSizeTensorTypes()` in the plugin build. Additionally, old Dropout opset 7–9 and 10–11 kernel registrations were moved from `identity_op.cc` to `nn/dropout.cc` so that each op's registrations live in that op's own source file.
- A few tensor kernels (`pad.cc`, `tile.cc`, `unsqueeze.cc`) still contain localized plugin guards where adapter and framework paths have not fully converged. Recent cleanup removed the plugin-only branches from `upsample.*`, `space_depth_ops.h`, and `scatter_nd.*` by moving reusable logic into shared adapter-safe helpers and by adding allocator access to `ep::adapter::OpKernelInfo`.

The broad trend remains positive: most operator-level plugin conditionals were removed by moving reusable CPU/helper logic into shared headers and by centralizing stream bridging in `CudaKernel` helpers.

---

## 9. Building

### 9.1 CMake Flag

The plugin is the default CUDA EP build when `onnxruntime_USE_CUDA=ON`. The `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN` option controls whether CUDA is built as the plugin EP or as the legacy in-tree provider:

- `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON` builds `onnxruntime_providers_cuda_plugin` and advertises it as `CUDAExecutionProvider`.
- `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=OFF` builds the legacy source-built `onnxruntime_providers_cuda` provider.

```bash
sh build.sh --config Release --build_dir build/cuda --parallel --use_cuda \
    --cuda_version 12.8 --cuda_home /path/to/cuda \
    --cudnn_home /path/to/cudnn \
    --build_wheel --skip_tests \
    --cmake_generator Ninja \
    --enable_cuda_nhwc_ops \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="90"
```

### 9.2 Impact on Other Build Targets

The `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON` flag replaces the legacy CUDA provider target with the plugin target. It:

1. Adds the `onnxruntime_providers_cuda_plugin` CMake target, whose native output uses the canonical CUDA provider filename (`libonnxruntime_providers_cuda.so` / `onnxruntime_providers_cuda.dll`)
2. Skips the legacy `onnxruntime_providers_cuda` target and CUDA EP internal unit-test library
3. Copies the plugin library into Python and Java package outputs when those packages are built
4. Appends `"cuda-plugin-ep=1"` to the build info string

Use `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=OFF` when you need to build the original in-tree CUDA EP from source.

### 9.3 Plugin Independence

The plugin build's `libonnxruntime_providers_cuda.so` is **fully self-contained**. It does not depend on `libonnxruntime_providers_shared.so` at load time. It statically links against `onnxruntime_framework`, `onnxruntime_graph`, `onnxruntime_common`, `onnxruntime_mlas`, `onnxruntime_flatbuffers`, and links dynamically against CUDA (`cudart`, `cublas`, `cublasLt`, `cufft`) and protobuf. cuDNN is loaded lazily only when enabled and available at runtime. Communication with the ORT runtime happens exclusively through the C API (`OrtApi`/`OrtEpApi`) passed at load time.

### 9.4 Build Outputs

After a successful build with the plugin flag ON, `build/cuda/Release/` contains:

| File | Description |
|------|-------------|
| `libonnxruntime_providers.a` | CPU provider (static, linked into main binary) |
| `libonnxruntime_providers_shared.so` | Shared provider bridge (for in-tree CUDA EP) |
| `libonnxruntime_providers_cuda.so` | Plugin CUDA EP when `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`; in-tree CUDA EP when it is OFF |

### 9.5 Deployment

To use the plugin EP with a bundled ONNX Runtime Python package, copy the plugin library to the ORT Python package's `capi/` directory using the canonical CUDA provider filename:

```bash
cp build/cuda/Release/libonnxruntime_providers_cuda.so \
   $(python -c "import onnxruntime; print(onnxruntime.__path__[0])")/capi/
```

If the package build info contains `cuda-plugin-ep=1`, importing `onnxruntime` auto-registers that bundled library as `CUDAExecutionProvider`. `onnxruntime.preload_dlls(...)` also retries bundled plugin registration after loading CUDA/cuDNN DLLs, which is useful on Windows. Standalone plugin packages and native applications continue to load the same file explicitly through `register_execution_provider_library()` / `RegisterExecutionProviderLibrary()` before creating sessions.

---

## 10. Testing

### 10.1 Test Script

`onnxruntime/test/python/transformers/test_cuda_plugin_ep.py` provides the current focused plugin validation flow:

| Category | What It Tests |
|----------|---------------|
| Registration | Dynamic loading via `register_execution_provider_library()` and EP device discovery (Add, MatMul, Gemm, Conv) |
| Provider options | Valid option parsing, invalid device rejection, multi-device selection |
| NHWC | NHWC-requested sessions: Conv, BatchNorm, MaxPool, AveragePool. These validate correctness under `prefer_nhwc` and require plugin-backed NHWC execution to succeed; they are the focused regression suite for the plugin NHWC path |
| Tensor ops | Reshape, Split, Concat, Gather, Unsqueeze, Tile, Pad, Slice, Transpose, Cast, Where, Flatten, ArgMax, TopK, Trilu, NonZero |
| Math ops | Softmax, Relu, Sigmoid, Tanh, Einsum (single and batched) |
| Reduce | ReduceMean, ReduceSum |
| Space/depth | SpaceToDepth, DepthToSpace, Upsample |
| Shape ops | CumSum, ConstantOfShape, Resize, Sum (variadic) |
| Normalization | LayerNormalization, InstanceNormalization |
| Conv | ConvTranspose |
| Scatter/gather | GatherND, ScatterElements, OneHot |
| Spatial | GridSample |
| Contrib ops | FastGelu, Gelu, BiasGelu, SkipLayerNorm, BiasDropout, FusedMatMul |
| Dropout | Dropout opset 7 and opset 10 — verifies registrations moved to `dropout.cc` |
| Quantization | DequantizeLinear / QuantizeLinear opset 21 — exercises `TWO_TYPED_KERNEL_EX` adapter macro; MatMulInteger |
| GatherBlockQuantized | Contrib `GatherBlockQuantized` — exercises `THREE_TYPED_KERNEL_EX` adapter macro |
| Identity | Identity opset 13 and opset 25 — re-enabled op with `TensorSeq` path guarded |
| Crop | Crop (opset 1) — previously excluded contrib op, now re-enabled |
| Memcpy | Explicit `MemcpyFromHost` and `MemcpyToHost` standalone tests to ensure copy ops are dispatched |
| CUDA Graph | Capture/replay with default arena (`test_cuda_graph_capture_and_replay`, `test_cuda_graph_add_model`), in-place input update after capture (`test_cuda_graph_replay_with_updated_input`), CUDA native mempool allocator (`test_cuda_graph_with_mempool`), and multiple annotation IDs (`test_cuda_graph_annotation_id`) |
| IOBinding / Sync | IOBinding-based tests (Add, MatMul) that bind CPU inputs and CUDA outputs to exercise `OrtEp::Sync` and `OrtEp::CreateSyncStreamForDevice` |
| Key-ops probe | Session-based probing that all key ops are assigned to `CUDAExecutionProvider` |

### 10.2 Running Tests

After building and deploying the plugin (see [Section 9.5](#95-deployment)), use the shared test instructions in [QUICK_START.md](QUICK_START.md#running-tests). That section is the source of truth for prerequisites, `ORT_CUDA_PLUGIN_PATH`, and platform-specific test commands.

### 10.3 Parity Report

`tools/ci_build/cuda_plugin_parity_report.py` generates both static and runtime parity reports:

- **Static mode** (default): Parses CMake exclusion patterns and kernel registration macros from source to compare what the plugin build includes vs. the bundled CUDA EP.
  ```bash
  python tools/ci_build/cuda_plugin_parity_report.py
  ```
- **Runtime mode** (`--runtime`): Uses the pybind `get_registered_ep_kernel_defs()` API (added in `onnxruntime_pybind_schema.cc`) to query actual kernel registries from both the bundled and plugin EPs, providing an accurate comparison of registered op/domain/version/type-constraint coverage.
  ```bash
  python tools/ci_build/cuda_plugin_parity_report.py --runtime [--plugin-ep-lib /path/to/plugin.so]
  ```

The runtime API (`get_registered_ep_kernel_defs(ep_name)`) creates a temporary EP factory and EP instance for the named EP, iterates its kernel registry, and returns `KernelDef` objects with `op_name`, `domain`, `version_range`, `provider`, and `type_constraints` fields.

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

### 11.4 If the kernel uses stream helpers

Prefer the shared helpers in `CudaKernel` instead of introducing new plugin-only stream shims:

- If the code only needs a raw CUDA stream, use `Stream(ctx)`.
- If the shared helper API accepts the adapter's opaque stream handle, use `GetComputeStream(ctx)`.
- If framework-style helper code still expects `onnxruntime::Stream*`, use `GetOrtStream(ctx)`.
- Prefer `GetCublasHandle(ctx)`, `GetCudnnHandle(ctx)`, and `GetCublasLtHandle(ctx)` over re-discovering handles from the stream manually.

### 11.5 If the kernel uses handle accessors

Use the plugin-compatible overloads already in `CudaKernel`:

```cpp
// Instead of:   GetCublasHandle(ctx->GetComputeStream())
// Use:          GetCublasHandle(ctx)  // or GetCublasHandle(Stream(ctx))
```

### 11.6 If OpSchema access is available — schema-validated type constraints

With the `OrtEpApi` OpSchema APIs (ORT ≥ 1.25, PR #27713), the plugin can validate or derive type constraint names at kernel registration time rather than relying solely on hard-coded strings.

#### 11.6.1 Validation mode (recommended first step)

Add a debug-mode validation pass in `CreateCudaKernelRegistry()` that runs after all kernels are collected from `PluginKernelCollector`. For each registered kernel, look up its `OrtOpSchema` and verify that every type constraint name used in the `KernelDef` actually appears in the schema's type constraint map:

```cpp
// In CreateCudaKernelRegistry(), after building the registry:
#ifndef NDEBUG
for (auto build_fn : entries) {
  auto info = build_fn();
  if (info.kernel_def == nullptr) continue;

  // Retrieve the op name, domain, and since_version from the KernelDef.
  const char* op_name = info.kernel_def->GetOpName();
  const char* domain = info.kernel_def->GetDomain();
  int since_version = info.kernel_def->GetSinceVersion();

  // Look up the ONNX schema from ORT's global registry.
  Ort::ConstOpSchema schema = Ort::GetOpSchema(op_name, since_version, domain);
  if (!schema) continue;  // contrib/internal ops may not have an ONNX schema

  // Validate each type constraint name against the schema.
  for (const auto& [constraint_name, types] : info.kernel_def->GetTypeConstraints()) {
    if (!schema.HasTypeConstraint(constraint_name.c_str())) {
      LOGS_DEFAULT(WARNING) << "Plugin kernel " << op_name
                            << ": type constraint '" << constraint_name
                            << "' not found in OpSchema (domain=" << domain
                            << ", version=" << since_version << ")";
    }
  }
}
#endif
```

This catches hard-to-debug kernel-matching failures caused by constraint name typos or opset version drift.

#### 11.6.2 Schema-driven constraint helper (future)

A `KernelDefBuilder` extension could derive constraint names from the schema automatically:

```cpp
/// Look up the type constraint string for a given input index from the OpSchema.
/// Falls back to the provided default if the schema is not found (e.g., contrib ops).
inline const char* GetInputTypeConstraintName(
    const char* op_name, int opset_version, const char* domain, size_t input_index,
    const char* fallback = "T") {
  Ort::ConstOpSchema schema = Ort::GetOpSchema(op_name, opset_version, domain);
  if (!schema || input_index >= schema.GetNumInputs()) return fallback;
  // Cache the result to avoid repeated lookups for typed kernel variants.
  static thread_local std::string cached_name;
  cached_name = schema.GetInputTypeStr(input_index);
  return cached_name.c_str();
}
```

This is a quality-of-life improvement rather than a required change — the existing hard-coded constraint names are correct for all currently registered kernels.

---

## 12. Memory Arena Integration

The CUDA plugin EP now includes a full BFC-style arena (`CudaArenaAllocator` / `ArenaImpl`) and a CUDA native mempool allocator (`CudaMempoolOrtAllocator`), both residing inside the plugin library. The detailed design — factory lifecycle, per-device cache, stream integration, arena config flow, and the `CudaMempoolArena` migration — is documented in [arena_allocator_migration_design.md](arena_allocator_migration_design.md).

**ORT core integration:** Plugin arenas implement `OrtAllocator::Shrink` (added in ORT API version 25). When ORT core detects a non-null `Shrink` function pointer on the returned `OrtAllocator*`, it wraps the allocator as `IArenaImplWrappingOrtAllocator` (an `IArena`). This makes the plugin arena visible to session-level arena management — `InferenceSession::ShrinkMemoryArenas()`, `ValidateAndParseShrinkArenaString()`, `DeviceStreamCollection::ReleaseSingleStreamBuffers()` — through the standard `IArena::SafeArenaCast()` / `AsArena()` virtual method, without requiring RTTI.

**Key files introduced:**

| File | Purpose |
|------|---------|
| `plugin/cuda_arena.h` | `ArenaConfig`, `ArenaImpl` (BFC arena), `CudaArenaAllocator` (`OrtAllocator` wrapper) |
| `plugin/cuda_arena.cc` | Arena implementation: bins, chunks, regions, stream-aware alloc, `Shrink()`, `GetStats()` |
| `plugin/cuda_mempool_allocator_plugin.h` | `CudaMempoolOrtAllocator` — wraps CUDA native mempool behind `OrtAllocator` |
| `plugin/cuda_mempool_allocator_plugin.cc` | Mempool implementation: `cudaMallocFromPoolAsync`/`cudaFreeAsync`, pool lifecycle, `Shrink()` via `cudaMemPoolTrimTo` |
| `core/session/allocator_adapters.h` | `IArenaImplWrappingOrtAllocator` — wraps plugin `OrtAllocator*` with `Shrink` as `IArena` |
| `core/session/allocator_adapters.cc` | Adapter implementation; `GetStatsFromOrtAllocator()` helper; `kOrtAllocatorShrinkMinVersion` |

---

## 13. File Layout

```
onnxruntime/core/providers/cuda/plugin/
├── cuda_kernel_adapter.h        # CudaKernel base, macros, CPU shims (force-included)
├── cuda_ep.h / .cc              # CudaEp : OrtEp implementation (GetCapability, Sync, CreateSyncStreamForDevice)
├── cuda_ep_factory.h / .cc      # CudaEpFactory : OrtEpFactory (arena lifecycle, per-device cache)
├── cuda_plugin_ep.cc            # DLL entry points (CreateEpFactories/ReleaseEpFactory)
├── cuda_plugin_ep_symbols.def   # Windows DLL export definitions
├── cuda_plugin_kernels.h / .cu  # Kernel registry creation
├── cuda_stream_plugin.h / .cc   # CudaSyncStream (handles, notifications, arena chunk reset)
├── cuda_allocator_plugin.h / .cc    # Device/pinned raw allocators (CudaAllocatorBase hierarchy)
├── cuda_arena.h / .cc           # BFC arena (ArenaConfig, ArenaImpl, CudaArenaAllocator)
├── cuda_mempool_allocator_plugin.h / .cc  # CUDA native mempool allocator (CudaMempoolOrtAllocator)
├── cuda_data_transfer_plugin.h / .cc # GPU↔CPU data transfer
├── cuda_memcpy_plugin.cc        # MemcpyFromHost/MemcpyToHost standalone kernels
├── cuda_controlflow_plugin.h / .cc / .cu  # If/Loop/Scan wrappers
├── cuda_plugin_utils.h          # Common macros, error handling
└── provider_api_shims.cc        # Reimplemented utility functions

onnxruntime/core/session/
├── allocator_adapters.h / .cc   # OrtAllocator↔IAllocator/IArena bidirectional adapters
│                                # (IAllocatorImplWrappingOrtAllocator, IArenaImplWrappingOrtAllocator,
│                                #  OrtAllocatorImplWrappingIAllocator)
└── ...

include/onnxruntime/core/framework/
├── allocator.h                  # IAllocator (AsArena virtual), IArena (Shrink, SafeArenaCast)
└── ...

include/onnxruntime/ep/
├── README.md                    # EP adapter layer overview
├── adapters.h                   # Master include + type aliasing (force-included)
├── api.h                        # ORT C API includes
├── common.h                     # EP common utilities
├── get_capability_utils.h       # GetCapability helper utilities
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

## 14. Profiling and Observability

The CUDA plugin EP implements the `OrtEpProfilerImpl` interface (introduced in ORT 1.25 via [PR #27649](https://github.com/microsoft/onnxruntime/pull/27649)) to participate in ORT's profiling system. When profiling is enabled, GPU kernel executions (CUDA kernels, memory copies) captured by NVIDIA CUPTI appear alongside ORT's CPU-side events in the profiling output.

### 14.1 Architecture

The profiling stack has three layers:

1. **ORT Core** (`Profiler` in `profiler.cc`) — drives the profiling lifecycle. It calls `PluginExecutionProvider::GetProfiler()`, which invokes `OrtEp::CreateProfiler` on the plugin and wraps the returned `OrtEpProfilerImpl` in a `PluginEpProfiler` bridge.
2. **Bridge** (`PluginEpProfiler` in `ep_event_profiling.cc`) — adapts the C++ `EpProfiler` interface to the C `OrtEpProfilerImpl` callbacks. It handles clock synchronization (provides an epoch-independent offset in `StartProfiling`) and converts relative ORT event IDs to absolute epoch-based correlation IDs for `StartEvent`/`StopEvent`.
3. **Plugin-side profiler** (`CudaPluginEpProfiler` in `cuda_profiler_plugin.h/.cc`) — implements `OrtEpProfilerImpl` inside the plugin DLL. Delegates to `CUPTIManager` for GPU activity tracing.

```
ORT Profiler
  └─ PluginEpProfiler (bridge, in ORT core)
       └─ OrtEpProfilerImpl callbacks (C API boundary)
            └─ CudaPluginEpProfiler (in plugin DLL)
                 └─ CUPTIManager singleton (in plugin DLL)
                      └─ CUPTI activity APIs (GPU tracing)
```

### 14.2 CUPTI Integration

The plugin DLL links `CUDA::cupti` and compiles `cupti_manager.cc` when `onnxruntime_ENABLE_CUDA_PROFILING` is ON. The `CUPTIManager` singleton lives inside the plugin DLL, isolated from any in-tree CUDA EP in the same process. This is the expected isolation model for plugin EPs.

CUPTI activities enabled:
- `CUPTI_ACTIVITY_KIND_RUNTIME` — CUDA runtime API calls
- `CUPTI_ACTIVITY_KIND_DRIVER` — CUDA driver API calls
- `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` — GPU kernel execution
- `CUPTI_ACTIVITY_KIND_MEMCPY` — device memory transfers
- `CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION` — maps GPU activities to ORT event correlation IDs

### 14.3 Correlation ID Flow

The plugin API's `StartEvent`/`StopEvent` receive **absolute epoch-based** correlation IDs (converted by the `PluginEpProfiler` bridge from ORT's relative event IDs). These are pushed directly to CUPTI's external correlation stack via `cuptiActivityPushExternalCorrelationId`, allowing CUPTI to tag GPU activities with the corresponding ORT event. When `StopEvent` is called, the correlation ID is popped. This matches the pattern used by the in-tree CUDA EP's `GPUTracerManager::PushCorrelation`/`PopCorrelation`.

### 14.4 Event Collection (EndProfiling)

When ORT calls `EndProfiling`:
1. CUPTI activity buffers are flushed (`cuptiActivityFlushAll`).
2. GPU activity records are processed — kernel names, timestamps, durations, and stream/grid metadata are extracted.
3. Events are converted to `Ort::ProfilingEvent` instances with `OrtProfilingEventCategory_KERNEL`.
4. Events are appended to the `OrtProfilingEventsContainer` via `AddEvents`.

The plugin does **not** perform the post-hoc merge/sort that the in-tree `GPUProfilerBase::EndProfiling` does. The plugin API is append-only, and the `PluginEpProfiler` bridge on the ORT side likewise appends EP events to ORT's profiling event collection without merge/sort by timestamp or correlation ID. Any ordering or interleaving into a global timeline is handled by downstream trace consumers.

### 14.5 Design Differences from In-Tree CUDA EP Profiler

| Aspect | In-tree CUDA EP | CUDA Plugin EP |
|--------|----------------|----------------|
| Event merge | `GPUProfilerBase::MergeEvents` interleaves GPU events into ORT's array (has known sort-order bug) | Append-only; ORT-side bridge appends only, and trace consumers handle ordering |
| Correlation IDs | Relative → absolute conversion in `GPUTracerManager::PushCorrelation` | Bridge provides absolute IDs directly; plugin pushes to CUPTI as-is |
| `StopEvent` metadata | Ignored (just pops correlation) | ORT event metadata available; currently unused, can annotate GPU events in future |
| GPU→ORT event linkage | Implicit via CUPTI external correlation IDs merged into timeline | GPU events carry only CUPTI metadata (`stream`, `grid_*`, `block_*`); no ORT correlation or parent identifier is attached. Downstream consumers must relate GPU kernels to ORT nodes via timestamp proximity. This is a known limitation; future work may attach `correlation_id` or parent event name via `StopEvent`'s `OrtProfilingEvent` parameter |
| Singleton scope | Process-wide `CUPTIManager` in main ORT DLL | DLL-local `CUPTIManager` in plugin (process isolation) |

### 14.6 Build Configuration

CUPTI profiling is conditional:
- **CMake flag**: `onnxruntime_ENABLE_CUDA_PROFILING=ON`
- **Compile definition**: `ENABLE_CUDA_PROFILING` added to the plugin target
- **Link**: `CUDA::cupti` linked to `onnxruntime_providers_cuda_plugin`
- **Source**: `cupti_manager.cc` compiled into the plugin

When profiling is disabled (default), `CudaEp::CreateProfiler` is set to `nullptr` and no CUPTI code is compiled.

### 14.7 Files

| File | Role |
|------|------|
| `plugin/cuda_profiler_plugin.h` | `CudaPluginEpProfiler` struct definition |
| `plugin/cuda_profiler_plugin.cc` | Profiler callback implementations |
| `plugin/cuda_ep.h` | `CreateProfilerImpl` declaration |
| `plugin/cuda_ep.cc` | `CreateProfiler` callback wiring |
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | Conditional CUPTI linkage |

---

## 15. Future Work

1. **Remaining stream/adapter parity for framework-style `Stream*` consumers** — Much of the broad `Stream*` gap has already been addressed: the plugin adapter now provides an `OrtStreamAdapter` / `PluginStreamShim` path for framework-style `Stream*` call sites, FFT is included, and quantization/diffusion kernels are no longer excluded as a class. Remaining work is narrower:

   - Continue using `Stream(context)` / `GetOrtStream(context)` patterns for migrated kernels rather than adding raw-stream-only forks.
   - Audit still-excluded directories that require more than a stream handle: `contrib_ops/cuda/llm/*`, `contrib_ops/cuda/transformers/*`, and `contrib_ops/cuda/collective/*`.
   - For each re-inclusion pass, add or extend focused plugin tests before removing the CMake exclusion.

2. **Contrib LLM migration pass** — Still open. The core CUDA LLM attention path is now adapter-safe, but `contrib_ops/cuda/llm/*` remains excluded in `cmake/onnxruntime_providers_cuda_plugin.cmake`. The remaining work is a dedicated contrib-LLM adapter pass: resolve any plugin build failures under `ORT_USE_EP_API_ADAPTERS`, keep the normal stream/scratch-buffer helpers, remove the `contrib_ops/cuda/llm/*` CMake filters, and add focused tests or parity-report coverage for the first re-included kernels.

3. **Tunable ops** — Implement a plugin-side `ITuningContext` and remove the `ORT_USE_EP_API_ADAPTERS` guards in `matmul.cc`/`gemm.cc` so the plugin can recover runtime kernel selection and profiling-based tuning behavior.

4. **TensorSeq and additional C API coverage** — Add enough sequence/tensor-sequence support to unblock `sequence_op.cc` (the last remaining TensorSeq-dependent file), and extend the ORT C API where needed for remaining framework-style attribute accessors such as string-array attributes used by RNN kernels. Note: `identity_op.cc` is now included in the plugin build — its TensorSeq code path is guarded by `#ifndef BUILD_CUDA_EP_AS_PLUGIN` and opset 14+ registrations use `AllFixedSizeTensorTypes()` (Tensor-only) instead of `AllFixedSizeTensorAndSequenceTensorTypes()`.

5. **Remaining contrib exclusions** — Remaining contrib exclusions are: `shrunken_gather.cc` (training), `transformers/*` (subgraph), `aten_ops/*` (ATen), `collective/*` (NCCL), and `llm/*` (contrib LLM pass).

6. **CI integration and targeted benchmarking** — Partially complete. Basic CUDA plugin build + `test_cuda_plugin_ep.py` coverage now exists in Linux and Windows plugin CI workflows. Remaining work is perf-oriented and feature-specific validation: add targeted benchmarks or perf gates for graph replay and allocator behavior, and extend CI once profiling and tunable-op support land.

7. **NHWC cleanup and hardening** — Partially complete. Runtime NHWC callbacks, second-pass capability handling for pre-assigned NHWC nodes, cached provider-config access, and focused Conv/BatchNormalization/Pool tests are in place. Remaining work is the cleanup described in [Section 5.3.1](#531-nhwc-layout-transformation-support): unify the conversion allowlist with the bundled CUDA EP, improve internal-domain kernel-miss diagnostics, and add stronger structural assertions that plugin-backed NHWC execution was actually selected.

8. **OpSchema-validated kernel registration after PR #27713** — PR #27713 has already landed, so the `OrtEpApi` and C++ wrappers for querying ONNX operator schemas are available (see [Section 3.5.1](#351-type-constraint-names-and-opschema-access)). The remaining work is plugin-side adoption:

    **A. Registration-time validation pass**

    Still open. Add a debug/diagnostic pass in `CreateCudaKernelRegistry()` that validates every registered kernel's type constraint names against the schema. This is the highest-value, lowest-risk change — it catches silent kernel-matching failures caused by constraint name drift without altering the registration flow. See [Section 11.6.1](#1161-validation-mode-recommended-first-step) for the implementation pattern.

    **B. NHWC internal-domain schema diagnostics**

    Still open. Extend the validation pass to cover `com.ms.internal.nhwc`-domain registrations. When kernel lookup fails for a rewritten NHWC node, the diagnostic can now report exactly which constraint name was expected vs. what the kernel registered, directly addressing the diagnostic requirement in [Section 5.3.1.3](#5313-nhwc-design-requirements).

    **C. Parity report enhancement**

    Still open. Update `cuda_plugin_parity_report.py` to use the schema API (via a small C++ test harness or Python ONNX bindings) to flag type-constraint mismatches between the plugin's registered kernels and the ONNX schema, in addition to the existing op-coverage comparison.

    **D. Schema-driven `KernelDefBuilder` helpers (longer term)**

    Still open and lower priority. Create a `KernelDefBuilder` helper that auto-derives constraint names from the schema instead of requiring hard-coded strings. This reduces maintenance burden when new opset versions introduce constraint name changes, but is lower priority than the validation pass since all current constraint names are correct.

    **E. Potential code locations for changes**

    | File | Change |
    |------|--------|
    | `cuda_plugin_kernels.cu` / `CreateCudaKernelRegistry()` | Add schema validation loop after kernel collection |
    | `cuda_kernel_adapter.h` | (Optional) Add schema-aware macro variant or post-registration hook |
    | `include/onnxruntime/ep/adapter/kernel_def_builder.h` | (Optional) Add schema-lookup helper for constraint names |
    | `cuda_ep.cc` / `GetCapabilityImpl()` | (Optional) Add schema-based diagnostic when `EpGraphSupportInfo_LookUpKernel` returns nullptr |
    | `test_cuda_plugin_ep.py` | Add a validation stage that exercises schema-validated registration |

9. **Resource accounting and annotation-based partitioning after PR #27595** — PR #27595 has already landed, so ORT now has framework-side resource accounting and layering annotations. The remaining CUDA plugin work is to bridge those capabilities through the plugin EP API and plugin capability implementation.

    **A. Resource accounting**

    `IResourceAccountant` lets an EP declare a resource budget (e.g., available VRAM) and have the partitioner stop assigning nodes once that budget is exhausted. The framework passes an `IResourceAccountant*` to `IExecutionProvider::GetCapability()`; the in-tree CUDA EP uses it to compute per-node estimated VRAM cost from initializer sizes.

    For plugin EPs, the `OrtEp::GetCapability` callback still has no mechanism to receive or report resource usage. `PluginExecutionProvider::GetCapability()` receives an `IResourceAccountant*`, but it currently leaves that parameter unused before calling the `OrtEp::GetCapability` callback. Two design options remain:

    - **Option A (preferred — ORT core change):** Add an `OrtEp` analogue of the current `IResourceAccountant` flow, such as resource-accounting helpers on `OrtEpGraphSupportInfo`. This would let the plugin request per-node resource budget during `CudaEp::GetCapabilityImpl()` without duplicating partitioner budget logic.

    - **Option B (plugin-side workaround):** Expose the VRAM threshold through a plugin-specific session option key. During `GetCapabilityImpl`, the plugin reads the threshold from the parsed config and performs its own initializer-size accounting using `OrtEp_GetNodeAttributes` / node-graph-view APIs already present in the `OrtEp` API surface. This avoids an ORT core change but duplicates budget-tracking logic.

    **B. Annotation-based layering**

    PR #27595 also introduced `layering_annotations` — node-level `"layer_ann"` metadata that routes nodes to specific EPs or CPU during partitioning. The expected model is that plugin EPs participate through the same `GetCapability` flow and therefore observe whatever node set ORT presents after applying layering rules. In practice that should mean no plugin-specific changes are needed to respect annotations that exclude nodes from the plugin. However, the plugin design should avoid depending on undocumented filtering details in the `OrtGraph*` contract. If the plugin EP itself needs to *read* layering annotations for internal decisions, or if the API needs to make filtered-vs-unfiltered graph semantics explicit, that would require new `OrtEp` API surface.

    Current known limitations to keep in future work:

    - The `cuda(...)` device selector matches the renamed CUDA plugin EP through the `CUDAExecutionProvider` name. Future work should keep this path covered as plugin device metadata evolves.
    - The `gpu:<index>(...)` selector is currently matched using `OrtHardwareDevice::device_id`. That field is not a stable CUDA ordinal and is not guaranteed to uniquely identify one physical GPU, so index-based layer assignment is unreliable for the CUDA plugin EP, especially on hosts with multiple similar NVIDIA GPUs.

    **Recommended action:** First add the plugin API bridge for resource accounting, then update `CudaEp::GetCapabilityImpl()` to request resource budget for candidate nodes when layer assignments exist. Until that bridge exists, the plugin can observe the filtered node set from ORT partitioning but cannot report resource consumption through the same `IResourceAccountant` flow as the in-tree CUDA EP.
