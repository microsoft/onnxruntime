# CUDA Graph Support for CUDA Plugin EP

## Design Overview

### Background

The CUDA Plugin EP is a standalone shared library (`libonnxruntime_providers_cuda.so`) that implements the OrtEp C API, allowing CUDA EP updates independent of ORT releases. CUDA graph capture/replay is a critical performance optimization that records a sequence of GPU operations into a graph, then replays it with minimal CPU overhead on subsequent runs.

The OrtEp C API (v1.26+) provides four graph-capture callbacks:

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `IsGraphCaptureEnabled` | `bool(const OrtEp*)` | Report whether graph capture is enabled |
| `IsGraphCaptured` | `bool(const OrtEp*, int graph_annotation_id)` | Check if a graph has been captured for a given annotation ID |
| `ReplayGraph` | `OrtStatus*(OrtEp*, int graph_annotation_id)` | Launch a previously captured graph |
| `GetGraphCaptureNodeAssignmentPolicy` | `OrtGraphCaptureNodeAssignmentPolicy(const OrtEp*)` | Specify validation strictness for node assignment |

These are supplemented by the existing `OnRunStart` / `OnRunEnd` lifecycle callbacks that drive the capture workflow.

### Architecture

```
Session::Run()
  │
  ├─ Run 1..N (warmup): OnRunStart → kernel dispatch → OnRunEnd (increment counter)
  │
  ├─ Run N+1 (capture):  OnRunStart → cudaStreamBeginCapture → kernel dispatch
  │                        → OnRunEnd → cudaStreamEndCapture → cudaGraphInstantiate → Replay
  │
  └─ Run N+2+ (replay):  IsGraphCaptured() → true → ReplayGraph() → cudaGraphLaunch
                          (OnRunStart/OnRunEnd are NOT called during replay)
```

**Key design choices:**

- Each thread gets its own dedicated graph `cudaStream_t`, `CudaGraphManager`, and capture bookkeeping for the EP instance. `CudaSyncStream::InitHandlesWithExternalStream()` wraps the thread's graph stream so graph capture sees the same stream as kernels. The manager stores captured `cudaGraphExec_t` executables keyed by annotation ID, allowing multiple graphs (e.g., different input shapes) for that thread.
- When a `user_compute_stream` is supplied together with graph capture, the per-thread context adopts that user-owned stream as its graph stream instead of creating one, so capture/replay run on the same stream the caller drives. The context records that it does not own the stream and never destroys it. See [User Compute Stream + CUDA Graph](#user-compute-stream--cuda-graph).
- Warm-up runs (default: 2) allow memory allocations to stabilize before capture begins.
- Graph annotation IDs are parsed from `OrtRunOptions` key `"gpu_graph_id"`. ID `-1` skips capture; `0` is the default.

### New Components

- **`CudaGraphSet`** — Hash map storage for `cudaGraphExec_t`, keyed by annotation ID. Owns the CUDA graph exec resources.
- **`CudaGraphManager`** — Orchestrates capture lifecycle: `CaptureBegin()`, `CaptureEnd()`, `Replay()`, warm-up tracking via `IncrementRegularRunCount()` / `IsGraphCaptureAllowed()`.
- **`CudaEp::PerThreadContext`** — Per-thread owner for the graph stream, `CudaGraphManager`, and the pre-capture free-memory watermark. The context is owned by a thread-local cache keyed by `CudaEp*`, so it is destroyed automatically when that thread exits. `CudaEp` keeps weak references to live thread-local cache maps only so it can erase its entry during EP teardown, and it prunes expired cache-map references while creating new contexts.
- **`CudaSyncStream::InitHandlesWithExternalStream()`** — Wraps an external (non-owned) `cudaStream_t` for registration/lifecycle tracking. Migrated kernels bind cuBLAS/cuDNN/cuBLASLt through thread-local fallback handles at dispatch time when the wrapper does not own library handles.

### Config Options

| Option Key | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep.cuda.enable_cuda_graph` | bool | false | Enable CUDA graph capture/replay |
| `ep.cuda.min_num_runs_before_cuda_graph_capture` | int | 2 | Warmup runs before capture |

Legacy flat alias `enable_cuda_graph` is also supported.

The provider option `user_compute_stream` (a `cudaStream_t` passed as a pointer) may be combined with `enable_cuda_graph`. See [User Compute Stream + CUDA Graph](#user-compute-stream--cuda-graph).

---

## User Compute Stream + CUDA Graph

A caller can supply its own CUDA stream through the `user_compute_stream` provider option and enable CUDA graph capture at the same time. This combination was previously rejected with `ORT_INVALID_ARGUMENT`; it is now supported and matches the bundled (non-plugin) CUDA EP.

When both options are set:

- `CudaEpFactory::CreateEpImpl` no longer rejects the pair. Setting `user_compute_stream` still forces unified-stream mode (matching the bundled EP).
- `CudaEp::CreateSyncStreamForDeviceImpl` wraps the user stream via `InitHandlesWithUserStream()`, attaching full cuBLAS/cuDNN/cuBLASLt handles to it.
- `CudaEp::GetPerThreadContext()` builds the thread's `PerThreadContext` around the user stream (`external_graph_stream`) instead of creating an EP-owned graph stream. Capture and replay therefore run on the same stream the kernels are issued to.
- The context records `owns_graph_stream = false`, so it tears down captured graph execs on destruction but never calls `cudaStreamDestroy` on the user-owned stream. Stream lifetime stays with the caller.

Because the user supplies one stream, this mode is inherently single-stream; the per-thread graph isolation still applies if the same session is driven from multiple threads, but each thread must drive its own captures on the stream it provides.

### `user_compute_stream` is not limited to the CUDA graph case

A natural question when reading `GetPerThreadContext()` is why `use_external_stream` is gated on `has_user_compute_stream && enable_cuda_graph` — does that restrict a user compute stream to graph-enabled runs? It does not.

- A user compute stream is honored for kernels in **both** graph and non-graph runs. That happens in `CudaEp::CreateSyncStreamForDeviceImpl`, whose first branch wraps `config_.user_compute_stream` via `InitHandlesWithUserStream()` **independently of `enable_cuda_graph`**.
- The `enable_cuda_graph` term in `use_external_stream` only governs the `PerThreadContext`'s *graph stream*. `PerThreadContext` is a graph-capture-only object: `GetPerThreadContext()` is reached exclusively from the graph path (the `enable_cuda_graph` branch of `CreateSyncStreamForDeviceImpl`, `OnRunStart`/`OnRunEnd`, `IsGraphCaptured`, `ReplayGraph`). With graph disabled, no `PerThreadContext` is ever constructed, so its stream-ownership flag is irrelevant.
- The flag therefore answers a narrower question — *"should the per-thread capture/replay graph stream adopt (and not destroy) the user's stream?"* — which is only meaningful when a graph is actually being captured.

## Implementation Summary

### Files Changed

| File | Change |
|------|--------|
| `onnxruntime/core/providers/cuda/plugin/cuda_ep.cc` | Implemented graph capture callbacks (`OnRunStartImpl`, `OnRunEndImpl`, `IsGraphCaptureEnabledImpl`, `IsGraphCapturedImpl`, `ReplayGraphImpl`, `IsConcurrentRunSupportedImpl`), updated `CreateSyncStreamForDeviceImpl` to wrap a `user_compute_stream` or otherwise use the current thread's graph stream when graph capture is enabled, made `PerThreadContext` adopt the user stream as its (non-owned) graph stream when `user_compute_stream` + `enable_cuda_graph` are combined, added per-thread graph state, preserved `sync_stream` synchronization, and added a `cudaMemGetInfo` defensive allocation check |
| `onnxruntime/core/providers/cuda/plugin/cuda_ep.h` | Added `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` config fields, graph callback declarations, and a per-thread graph context cache |
| `onnxruntime/core/providers/cuda/plugin/cuda_graph_plugin.cc` | **NEW** — Complete `CudaGraphSet` and `CudaGraphManager` implementation |
| `onnxruntime/core/providers/cuda/plugin/cuda_graph_plugin.h` | **NEW** — Header for graph manager types and constants |
| `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.cc` | Added `InitHandlesWithExternalStream()`, updated destructor for `owns_stream_` |
| `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h` | Added `InitHandlesWithExternalStream()` declaration, `owns_stream_` member |
| `onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.cc` | Added config parsing for `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture`; removed the validation that rejected `user_compute_stream` + `enable_cuda_graph` (the combination is now supported) |
| `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h` | `CudaKernel::GetScratchBuffer` now allocates through `Info().GetAllocator()` (the EP arena) and stream-tags scratch chunks with the framework stream exposed by `KernelContext_GetSyncStream`, instead of issuing a raw `cudaMallocAsync`/`cudaMalloc` per call |
| `include/onnxruntime/ep/adapter/allocator.h` | Implemented `IAllocatorWrappingOrtAllocator::IsStreamAware`/`AllocOnStream` (previously `ORT_NOT_IMPLEMENTED`) so plugin adapters can forward stream-aware allocations when a framework stream is available |
| `include/onnxruntime/core/session/onnxruntime_c_api.h` | Added `KernelContext_GetSyncStream` so plugin kernels can obtain the framework `OrtSyncStream*` for stream-aware allocation bookkeeping while still using `KernelContext_GetGPUComputeStream` for raw CUDA work |
| `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` | Added `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy` callbacks and `OrtGraphCaptureNodeAssignmentPolicy` enum to `OrtEp` |
| `include/onnxruntime/core/framework/execution_provider.h` | Added `GetGraphCaptureNodeAssignmentPolicy()` virtual to `IExecutionProvider` |
| `onnxruntime/core/session/inference_session.cc` | Replaced hard-coded EP name list with policy-driven graph capture validation loop; added bounded recursion via `RunImpl()` with `kMaxGraphCaptureWarmupRuns`; graph-enabled runs now reacquire stream collections through ORT core's thread-affine pool across internal warm-up/capture recursion |
| `onnxruntime/core/framework/session_state.cc` | Sharded the `DeviceStreamCollection` cache by caller thread using per-thread lifetime tokens, so stream wrappers are only reused on the creating thread; added a fallback in the PrePack loop to resolve the kernel's default-memory allocator (`Info().GetAllocator()`) when the device-keyed initializer-allocator lookup returns null for a separately-registered plugin EP |
| `onnxruntime/core/framework/session_state.h` | Added thread-affine stream pool bucket state for `DeviceStreamCollection` reuse |
| `onnxruntime/core/session/inference_session.h` | Added `RunImpl()` private method and `kMaxGraphCaptureWarmupRuns` constant |
| `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc` | Added version-gated `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy` bridge implementations |
| `onnxruntime/core/providers/webgpu/ep/ep.cc` | Added graph capture callback delegation to underlying `IExecutionProvider` |

### Key Design Decisions

- **`GetGraphCaptureNodeAssignmentPolicy`**: Returns `ALLOW_CPU_FOR_SHAPES` — consistent with the non-plugin CUDA EP behavior and allows shape-inference nodes on CPU.
- **Thread safety**: Mutable graph state and graph streams are stored per thread. ORT core's `DeviceStreamCollection` cache is also thread-affine, so graph-enabled runs can recycle stream wrappers without exposing them to a different thread.
- **Scope**: Capture/replay pipeline plus allocator compatibility. Arena integration is complete — see the [Arena Allocator Integration](#arena-allocator-integration) section.
- **Callback assignment**: `IsGraphCaptureEnabled` and `GetGraphCaptureNodeAssignmentPolicy` are always set. `OnRunStart`, `OnRunEnd` are conditional on `enable_cuda_graph`. `IsGraphCaptured` and `ReplayGraph` are always set (return false/error when disabled).
- **Stream management**: `CreateSyncStreamForDevice` remains unconditional — it branches internally: it wraps a user-provided `user_compute_stream` (via `InitHandlesWithUserStream`) when one is set, otherwise uses the current thread's graph stream (via `InitHandlesWithExternalStream`) when graph capture is enabled, or creates an owned stream when both are disabled.
- **User compute stream + CUDA graph**: These options can now be combined (previously rejected at factory creation). When both are set, `CudaEp::GetPerThreadContext()` builds the `PerThreadContext` around the user's stream (`external_graph_stream`) so capture and replay run on the same stream the kernels use, and the context never destroys the user-owned stream (`owns_graph_stream = false`).
- **Run-end synchronization**: `OnRunEndImpl` honors the `sync_stream` flag without double-synchronizing replayed graphs, preserving the normal EP completion contract.
- **Stream collection reuse**: ORT core now recycles `DeviceStreamCollection` objects into a thread-affine session pool keyed by a per-thread lifetime token. Warm-up, capture, replay, and later user-visible `Run()` calls on the same thread can reuse the same stream wrappers, while dead-thread buckets are pruned before they can be reused by another thread.
- **Per-thread context lifecycle**: Thread-local caches hold the strong `PerThreadContext` references, so CUDA streams and captured graph executables are released when the owning thread exits. The EP tracks weak references to those cache maps to remove stale entries during EP destruction without keeping the contexts alive.

### Arena Allocator Integration

CUDA graph capture requires that all memory allocations happen during warmup, not during capture. The plugin arena allocator (PR #27931) is now landed and integrated with the graph capture path.

**Allocation-during-capture detection:**

- `OnRunStartImpl` records free GPU memory in the per-thread context via `cudaMemGetInfo` before `CaptureBegin`.
- `OnRunEndImpl` compares post-capture free memory in the same per-thread context. If it decreased, a warning is logged advising the user to increase `min_num_runs_before_cuda_graph_capture`.
- This `cudaMemGetInfo` check is retained as a last-line diagnostic after arena integration, because custom arena options, insufficient warm-up, or regressions can still surface allocation-during-capture issues.

**Arena integration details (now implemented):**

- Default CUDA device allocations come from the plugin-hosted arena (`CudaArenaAllocator`). During warmup runs, the arena grows to accommodate all needed chunks; during capture and replay, the same chunks are reused without `cudaMalloc` calls.
- Kernel scratch/workspace allocations (`CudaKernel::GetScratchBuffer`) also flow through the EP arena via `Info().GetAllocator()`, rather than issuing a fresh `cudaMallocAsync`/`cudaMalloc` per call. After warmup the arena has reached its steady-state working set, so the capture run serves every scratch request from an already-reserved chunk and the device free-memory footprint stays stable across the capture window. This is what makes the `cudaMemGetInfo` allocation-during-capture detector pass for graphs that use scratch buffers, and it matches the bundled CUDA EP (which also obtains scratch from `Info().GetAllocator()`). `GetScratchBuffer` stream-tags scratch chunks with the framework `OrtSyncStream*` exposed by `KernelContext_GetSyncStream`. The raw `cudaStream_t` from `KernelContext_GetGPUComputeStream` is still used for CUDA launches and library calls; the framework stream is used only for the arena's cross-stream reuse bookkeeping.
- When `arena.use_cuda_mempool=1` is configured, CUDA device allocations come from `CudaMempoolOrtAllocator`, which wraps `cudaMallocFromPoolAsync`/`cudaFreeAsync`. These async allocation/free operations are CUDA-graph-safe since CUDA 11.4+ and become part of the captured graph topology.
- Pinned allocations are also arena-backed, but remain non-stream-aware.
- The graph stream created by `CudaEp::PerThreadContext` flows through `CudaSyncStream::InitHandlesWithExternalStream()` so stream-aware arena allocation uses the same `cudaStream_t` during warm-up, capture, and replay.
- `CudaSyncStream::OnSessionRunEndImpl()` resets arena chunk-to-stream assignments via `factory_.ResetDeviceArenaChunksUsingStream()` at the end of each run, even for graph-enabled runs. `OnSessionRunEnd` executes before the stream collection is recycled into the current thread's pool bucket.
- The plugin allocator's `OrtMemoryInfo::alloc_type` stays as `OrtDeviceAllocator`; the arena remains opaque to ORT core.

### Concurrent Run Support

Concurrent `Session::Run()` is advertised by the CUDA plugin EP when the host ORT runtime exposes `KernelContext_GetSyncStream` and the session is not forced into EP-level unified-stream mode.

- `CudaEp::PerThreadContext` still owns graph stream, graph manager, warm-up run counts, and memory watermark state per thread. This keeps graph bookkeeping thread-local and avoids sharing captured graph executables across threads.
- Plugin kernels now obtain the framework `OrtSyncStream*` through `KernelContext_GetSyncStream` and use it only for scratch/workspace allocation bookkeeping. CUDA work still launches on the raw `cudaStream_t` from `KernelContext_GetGPUComputeStream`.
- Stream-tagged scratch chunks let the shared arena apply its normal cross-stream reuse rules for overlapping runs on different CUDA streams.
- When the negotiated ORT API version does not include `KernelContext_GetSyncStream`, `CudaKernel::GetScratchBuffer` falls back to a null stream tag and `CudaEp::IsConcurrentRunSupportedImpl()` returns false.

## Verification

1. Build and deploy the plugin using the instructions in [QUICK_START.md](QUICK_START.md#build-instructions) and [QUICK_START.md](QUICK_START.md#running-tests).
2. Run `onnxruntime/test/python/transformers/test_cuda_plugin_ep.py` as described in [QUICK_START.md](QUICK_START.md#running-tests).
3. The CUDA graph tests in that script validate:
   - `test_cuda_graph_capture_and_replay` — warmup + capture + replay with default arena
   - `test_cuda_graph_replay_with_updated_input` — in-place input update after graph capture
   - `test_cuda_graph_with_mempool` — graph capture with `arena.use_cuda_mempool=1`
   - `test_cuda_graph_annotation_id` — multiple graphs via `gpu_graph_id` run config
   - `test_cuda_graph_add_model` — graph capture with Add op (arena-backed)
4. `onnxruntime/test/providers/cuda/plugin/cuda_plugin_user_stream_graph_test.cc` is a C++ test (gated by `ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP`) covering `user_compute_stream` combined with `enable_cuda_graph`: it verifies session creation succeeds with both options set (regression for the removed validation), capture + replay on the user stream produce correct results, and replay after an in-place input update on the user stream is correct.

## Future Work

1. **Profiling integration**: CUDA graph replay currently bypasses the CUDA plugin EP profiler path because the CUDA plugin EP does not yet implement `OrtEp::CreateProfiler`. Wiring graph replay into that path is future work.
