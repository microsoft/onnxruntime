## Description

Ports graph capture/replay APIs (e.g., CUDA Graph) to the Plugin EP (`OrtEp`) C API so that plugin-based execution providers can participate in ORT-managed graph capture and replay.

### What changed

**New Plugin EP C API functions** (`onnxruntime_ep_c_api.h`):
- `OrtEp::IsGraphCaptureEnabled` — indicates whether the EP has graph capture enabled.
- `OrtEp::IsGraphCaptured` — indicates whether a graph has been captured for a given annotation ID.
- `OrtEp::ReplayGraph` — replays a previously captured graph.
- `OrtEp::GetGraphCaptureNodeAssignmentPolicy` — returns the node assignment validation policy for graph capture.

All four are optional (NULL defaults to safe behavior) and version-gated (`ort_version_supported >= 26`).
If `IsGraphCaptureEnabled` returns true, `IsGraphCaptured` and `ReplayGraph` must also be implemented;
otherwise `PluginExecutionProvider` logs a warning and disables graph capture for that EP.

**New `OrtGraphCaptureNodeAssignmentPolicy` enum** (`onnxruntime_ep_c_api.h`):
Replaces the hardcoded EP-name checks in `InferenceSession::Initialize()` with a policy-based approach:
- `ALL_NODES_ON_EP` — all nodes must be on the target EP (e.g., TensorRT).
- `ALLOW_CPU_FOR_SHAPES` — CPU nodes allowed for shape computation if no memcpy nodes exist (e.g., CUDA, WebGPU, DML).

**Refactored `InferenceSession` graph capture selection** (`inference_session.cc`):
- Removed the hardcoded `graph_support_ep_list` and per-EP `strcmp` checks.
- Now iterates over all registered EPs and uses `IsGraphCaptureEnabled()` + `GetGraphCaptureNodeAssignmentPolicy()` to select and validate the graph-capturing EP.
- `AreAllComputeNodesAssignedToCudaOrJsOrDmlEpWebGpuEp()` → generalized to `AreAllComputeNodesAssignedToEpOrCpu()`, which also requires at least one node on the target EP.
- `IExecutionProvider::GetGraphCaptureNodeAssignmentPolicy()` added to the base class (defaults to `ALL_NODES_ON_EP`).

**Bounded graph capture recursion** (`inference_session.cc/h`):
- `Run()` now delegates to `RunImpl()` with a `graph_capture_depth` parameter.
- Caps internal run attempts at `kMaxGraphCaptureRunAttempts = 8`, returning a clear error if the EP never reports `IsGraphCaptured() == true`.

**EP implementations**:
- **WebGPU plugin EP**: Fully implements all four graph capture APIs by forwarding to the underlying `IExecutionProvider`.
- **CUDA plugin EP**: Stubs with TODOs (returns disabled/not-implemented).
- **NvTensorRTRTX EP**: `IsGraphCaptureEnabled()` now returns `false` since this EP manages graph capture internally (not via ORT).

**C++ wrapper** (`onnxruntime_cxx_api.h` / `onnxruntime_cxx_inline.h`):
- Added `Ort::Env::CopyTensor()` convenience overload for copying a single tensor (wraps `CopyTensors` with `num_tensors=1`).

### Tests
- **`ep_plugin_provider_test.cc`**: Unit tests for each new `PluginExecutionProvider` graph capture method, including NULL function pointer defaults, version < 26 backward compatibility, and validation that `IsGraphCaptureEnabled()` returns false when `IsGraphCaptured` or `ReplayGraph` are NULL.
- **`test_graph_capture.cc`**: End-to-end test for WebGPU plugin EP graph capture/replay using IO binding (warm-up + capture run, then replay with different inputs).

### Motivation and Context

Previously, graph capture support was limited to a hardcoded list of EPs (`kCudaExecutionProvider`, `kTensorrtExecutionProvider`, `kJsExecutionProvider`, `kWebGpuExecutionProvider`, `kDmlExecutionProvider`) with EP-specific validation logic in `InferenceSession`. This made it impossible for plugin EPs to participate in ORT-managed graph capture/replay without modifying the core session code.

This PR makes graph capture/replay extensible to any EP, including out-of-tree plugin EPs, by exposing it through the `OrtEp` C API.
