# CUDA Graph Support for CUDA Plugin EP

## Design Overview

### Background

The CUDA Plugin EP is a standalone shared library (`libonnxruntime_providers_cuda_plugin.so`) that implements the OrtEp C API, allowing CUDA EP updates independent of ORT releases. CUDA graph capture/replay is a critical performance optimization that records a sequence of GPU operations into a graph, then replays it with minimal CPU overhead on subsequent runs.

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
- Warm-up runs (default: 2) allow memory allocations to stabilize before capture begins.
- Graph annotation IDs are parsed from `OrtRunOptions` key `"gpu_graph_id"`. ID `-1` skips capture; `0` is the default.

### New Components

- **`CudaGraphSet`** — Hash map storage for `cudaGraphExec_t`, keyed by annotation ID. Owns the CUDA graph exec resources.
- **`CudaGraphManager`** — Orchestrates capture lifecycle: `CaptureBegin()`, `CaptureEnd()`, `Replay()`, warm-up tracking via `IncrementRegularRunCount()` / `IsGraphCaptureAllowed()`.
- **`CudaEp::PerThreadContext`** — Per-thread owner for the graph stream, `CudaGraphManager`, and the pre-capture free-memory watermark. The context is owned by a thread-local cache keyed by `CudaEp*`, so it is destroyed automatically when that thread exits. `CudaEp` keeps weak references to live thread-local cache maps only so it can erase its entry during EP teardown, and it prunes expired cache-map references while creating new contexts.
- **`CudaSyncStream::InitHandlesWithExternalStream()`** — Wraps an external (non-owned) `cudaStream_t` with cuBLAS/cuDNN/cuBLASLt handles. Used so that kernel dispatches go through the EP's graph-capture stream.

### Config Options

| Option Key | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep.cudapluginexecutionprovider.enable_cuda_graph` | bool | false | Enable CUDA graph capture/replay |
| `ep.cudapluginexecutionprovider.min_num_runs_before_cuda_graph_capture` | int | 2 | Warmup runs before capture |

Legacy aliases `ep.cuda.enable_cuda_graph` and `enable_cuda_graph` are also supported.

---

## Implementation Summary

### Files Changed

| File | Change |
|------|--------|
| `onnxruntime/core/providers/cuda/plugin/cuda_ep.cc` | Implemented graph capture callbacks (`OnRunStartImpl`, `OnRunEndImpl`, `IsGraphCaptureEnabledImpl`, `IsGraphCapturedImpl`, `ReplayGraphImpl`, `IsConcurrentRunSupportedImpl`), updated `CreateSyncStreamForDeviceImpl` to use the current thread's graph stream when graph capture is enabled, added per-thread graph state, preserved `sync_stream` synchronization, and added a `cudaMemGetInfo` defensive allocation check |
| `onnxruntime/core/providers/cuda/plugin/cuda_ep.h` | Added `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` config fields, graph callback declarations, and a per-thread graph context cache |
| `onnxruntime/core/providers/cuda/plugin/cuda_graph_plugin.cc` | **NEW** — Complete `CudaGraphSet` and `CudaGraphManager` implementation |
| `onnxruntime/core/providers/cuda/plugin/cuda_graph_plugin.h` | **NEW** — Header for graph manager types and constants |
| `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.cc` | Added `InitHandlesWithExternalStream()`, updated destructor for `owns_stream_` |
| `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h` | Added `InitHandlesWithExternalStream()` declaration, `owns_stream_` member |
| `onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.cc` | Added config parsing for `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` |
| `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` | Added `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy` callbacks and `OrtGraphCaptureNodeAssignmentPolicy` enum to `OrtEp` |
| `include/onnxruntime/core/framework/execution_provider.h` | Added `GetGraphCaptureNodeAssignmentPolicy()` virtual to `IExecutionProvider` |
| `onnxruntime/core/session/inference_session.cc` | Replaced hard-coded EP name list with policy-driven graph capture validation loop; added bounded recursion via `RunImpl()` with `kMaxGraphCaptureWarmupRuns`; stream collection not recycled during graph capture |
| `onnxruntime/core/session/inference_session.h` | Added `RunImpl()` private method and `kMaxGraphCaptureWarmupRuns` constant |
| `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc` | Added version-gated `IsGraphCaptureEnabled`, `IsGraphCaptured`, `ReplayGraph`, `GetGraphCaptureNodeAssignmentPolicy` bridge implementations |
| In-tree EPs (CUDA, DML, JS, WebGPU) | Added `GetGraphCaptureNodeAssignmentPolicy()` override returning `ALLOW_CPU_FOR_SHAPES` |
| `onnxruntime/core/providers/webgpu/ep/ep.cc` | Added graph capture callback delegation to underlying `IExecutionProvider` |
| `onnxruntime/core/providers/nv_tensorrt_rtx/nv_execution_provider.cc` | Changed `IsGraphCaptureEnabled()` to return `false` (NvTRT RTX manages graph capture internally) |

### Key Design Decisions

- **`GetGraphCaptureNodeAssignmentPolicy`**: Returns `ALLOW_CPU_FOR_SHAPES` — consistent with the non-plugin CUDA EP behavior and allows shape-inference nodes on CPU.
- **Thread safety**: Mutable graph state and graph streams are stored per thread. CUDA graph mode reports concurrent `Session::Run()` as supported because overlapping runs no longer share graph IDs, run counts, capture state, replay state, or graph streams.
- **Scope**: Capture/replay pipeline plus allocator compatibility. The arena allocator work is expected to come from PR #27931; this graph path should integrate with that allocator after rebasing instead of reintroducing separate allocation behavior.
- **Callback assignment**: `IsGraphCaptureEnabled` and `GetGraphCaptureNodeAssignmentPolicy` are always set. `OnRunStart`, `OnRunEnd` are conditional on `enable_cuda_graph`. `IsGraphCaptured` and `ReplayGraph` are always set (return false/error when disabled).
- **Stream management**: `CreateSyncStreamForDevice` remains unconditional — it branches internally to use the current thread's graph stream (via `InitHandlesWithExternalStream`) when graph capture is enabled, or creates an owned stream when disabled.
- **Run-end synchronization**: `OnRunEndImpl` honors the `sync_stream` flag by synchronizing the graph stream after warm-up/capture bookkeeping, preserving the normal EP completion contract.
- **Stream collection reuse**: ORT does not recycle device stream collections while graph capture is enabled. This prevents a stream wrapper that points at one thread's graph stream from being reused by a later run on another thread.
- **Per-thread context lifecycle**: Thread-local caches hold the strong `PerThreadContext` references, so CUDA streams and captured graph executables are released when the owning thread exits. The EP tracks weak references to those cache maps to remove stale entries during EP destruction without keeping the contexts alive.

### Arena Allocator Integration

CUDA graph capture requires that all memory allocations happen during warmup, not during capture. Before PR #27931, the plugin's direct `cudaMalloc` allocator can allocate during capture. To detect this:

- `OnRunStartImpl` records free GPU memory in the per-thread context via `cudaMemGetInfo` before `CaptureBegin`.
- `OnRunEndImpl` compares post-capture free memory in the same per-thread context. If it decreased, a warning is logged advising the user to increase `min_num_runs_before_cuda_graph_capture`.

Assuming PR #27931 lands first, the plugin allocator path will be arena-backed:

- Default CUDA device allocations should come from the plugin-hosted arena (`CudaArenaAllocator`) rather than raw `cudaMalloc`/`cudaFree`.
- If `arena.use_cuda_mempool=1` is configured, CUDA device allocations should come from `CudaMempoolOrtAllocator`, which wraps the native CUDA mempool path.
- Pinned allocations are also arena-backed, but remain non-stream-aware.
- The graph stream created by `CudaEp::PerThreadContext` must continue to flow through `CudaSyncStream::InitHandlesWithExternalStream()` so stream-aware allocation can use the same `cudaStream_t` during warm-up, capture, and replay.
- Warm-up run count should be validated with arena enabled. The default of 2 is intended to let arena chunk creation and stream assignment settle before `cudaStreamBeginCapture`, but graph tests should cover both default arena and `arena.use_cuda_mempool=1`.
- The `cudaMemGetInfo` check should remain as a defensive diagnostic even after arena integration, because custom arena options or unusual model behavior can still surface allocation-during-capture regressions.

### Concurrent Run Support

Concurrent `Session::Run()` is supported with CUDA graph enabled by keeping capture/replay state thread-local:

- `CudaEp::PerThreadContext` owns the graph stream, graph manager, warm-up run counts, and memory watermark for the current thread.
- The current thread's cache owns the `PerThreadContext`; new threads get independent contexts, and exited threads release their contexts automatically.
- `CreateSyncStreamForDeviceImpl()` wraps the current thread's graph stream, so warm-up, capture, and replay all use the same stream for that thread.
- `CudaGraphManager::CaptureBegin()` uses `cudaStreamCaptureModeThreadLocal`, allowing overlapping capture scopes on different threads.
- ORT destroys graph-capture stream wrappers at the end of graph-enabled runs instead of recycling them into the session-wide stream collection pool. This avoids reusing a wrapper bound to one thread's graph stream on a different thread.
- `IsGraphCaptured()` and `ReplayGraph()` resolve the current thread's graph context. If a new thread runs a graph-enabled session for the first time, that thread performs its own warm-up and capture before replaying.

## Verification

1. Build: `./cuda_plugin.sh` — compiles with no errors
2. Test without graph: `./cuda_plugin.sh --test_plugin` — existing tests pass
3. Test with graph: Run `test_cuda_plugin_ep.py` with `enable_cuda_graph=1` — warmup + capture + replay succeeds

## Future Work

1. **Arena validation after PR #27931**: Rebase onto the plugin arena allocator work and validate CUDA graph capture with default `CudaArenaAllocator` and `arena.use_cuda_mempool=1`.
2. **Multi-graph with dynamic shapes**: The annotation ID system supports this, but testing with variable input shapes is needed.
3. **Profiling integration**: CUDA graph replay currently bypasses the profiler. Integration with the plugin profiler (when available) is future work.
