# WebGPU CPU and GPU timelines

Enable ONNX Runtime profiling through the native Node.js capture tool described below,
or through the usual profiling API in your application.

Open the resulting JSON trace in Perfetto or a Chrome trace viewer. GPU dispatch events use the same
microsecond time origin as the CPU events. On Windows D3D12 with bundled Dawn and an ORT-created device,
ORT uses `ID3D12CommandQueue::GetClockCalibration()` to correlate the GPU counter with the CPU's
QueryPerformanceCounter clock. This preserves the delay between CPU submission and GPU execution.
Calibration refreshes each inference run while retaining the profiler's time origin across runs.
It does not add a queue wait or change dispatch batching.

Each GPU event has a `clock_alignment` argument:

- `d3d12_clock_calibration`: native CPU/GPU clock correlation succeeded.
- `first_submit`: the first GPU event is anchored to CPU submission time. GPU durations and relative
  spacing are available, but this approximation cannot measure the initial GPU queueing delay.

The submission-time approximation remains in use for external Dawn API packages, custom devices or
proc tables, other backends/platforms, incompatible timestamp toggles, and failed native calibration.
GPU events require a device with timestamp-query support. Concurrent profiling of sessions sharing
a WebGPU context is unsupported.

WebGPU also records CPU-side `WebGPU::Encode`, `WebGPU::Submit`, `WebGPU::Wait`,
`WebGPU::StartProfiling` and `WebGPU::ProfileReadback` spans on the same CPU time origin.
These supplement the existing ORT model-run, executor and operator events without changing the core
profiler. They identify command preparation, submission, waits and profiling overhead. Nested spans
overlap: for example, a pipeline wait can be inside command encoding. Do not sum them as independent work.

Clock calibration, CPU event clocks/storage, timestamp writes and query resolve/readback run only
while profiling is active. Timestamp feature negotiation at device creation remains unchanged, so a
session can enable run-level profiling later. Profiling adds measurement and synchronization overhead;
profiled wall times should not be treated as normal inference throughput.
CPU timing gates are inline: disabled execution skips clock reads and event-recording calls and
allocates no event data.

## Capture and visualize a model with Node.js

The generic capture runner and dependency-free report generator require Node.js 20 or newer. Capture requires
an `onnxruntime-node` package built with native WebGPU and this change; the reporting tool needs no
ORT package. The default package path is this checkout's `js/node` directory. Use `--ort-module`
to select another local native Node package. A CPU-only package fails explicitly.

For source builds, follow the normal [Node.js binding build instructions](../js/node/README.md).
The existing ORT build toolchain is unchanged; capture, report generation and viewing require no
Python installation. An external Dawn API-only package still requires a host that supplies a
compatible Dawn proc table in the same process. A self-hosted capture needs bundled Dawn.

Describe every model input in `inputs.json`, with a tensor type, dimensions and either a flat data
array or `data_file` containing raw little-endian tensor bytes. File paths in the manifest are relative
to the manifest's directory. For example, an Abs model with float32 input `x` of shape `[2, 2]` uses:

```json
{
  "x": { "type": "float32", "dims": [2, 2], "data": [-1, 2, -3, 4] }
}
```

Supported numeric types are float16/32/64, signed/unsigned 8/16/32/64-bit integers and bool. Inline
float16 values are raw half-precision bit patterns. Use decimal strings for 64-bit integers outside
JavaScript's safe integer range; bool values are 0 or 1. Scalar dimensions are `[]`. Strings, sequences
and non-tensor values are unsupported. Inputs must exactly match the model's required input names.

```powershell
node onnxruntime/test/webgpu/profile_model.mjs model.onnx --inputs inputs.json --output-dir build/profile --warmup 1 --repeat 3
node onnxruntime/test/webgpu/visualize_profile.mjs <generated-trace.json> --metadata build/profile/capture.json --output build/profile/timeline.html
```

The capture output directory must be new. It contains the original ORT JSON trace and `capture.json`
with run timings, warmup labels, model/input hashes, selected module/version/backend/session settings,
and first-measured-run output hashes/shapes/samples. Hashing and output validation happen outside the
measured interval. `wall_ms` includes Node scheduling, inference and transfer of outputs to CPU;
initialization, output checks/disposal and input-file loading are excluded. This runner repeats fixed
inputs, uses CPU input/output tensors and disables CPU fallback. It does not change model graphs,
provider defaults, batching or graph-capture settings. `--intra-op-threads` defaults to 1; `--backend`
selects D3D12 (default), Vulkan or Metal. Only Windows D3D12 was executed locally.

Generate a separate capture with identical inputs/settings and `--no-profile` for ordinary wall times:

```powershell
node onnxruntime/test/webgpu/profile_model.mjs model.onnx --inputs inputs.json --output-dir build/baseline --warmup 1 --repeat 3 --no-profile
node onnxruntime/test/webgpu/visualize_profile.mjs <generated-trace.json> --metadata build/profile/capture.json --baseline build/baseline/capture.json --output build/profile/timeline.html
```

Baseline matching checks the model-file hash, input contents/types/shapes, ORT version/module, backend,
workload and session options. Warmup is excluded from baseline means. Keep external model weight files
and machine conditions identical; these are not verified by those hashes. With profiling disabled,
the runner neither enables nor ends profiling and creates no trace. This tool adds no production
runtime work.

The self-contained HTML viewer has aligned CPU/GPU lanes, run and warmup filters, search, an event
inspector, zoom/pan, program summaries and PNG/JSON exports. Filters use the stages present in the
capture metadata; the generic runner labels runs `inference`. Open it in a current Chromium browser.
If local-file viewing is restricted, serve the output with a local HTTP server. For other ORT traces,
omit `--metadata`; GPU events must contain WebGPU `clock_alignment` information. Profiling adds query,
trace-collection and synchronization overhead, so its timeline gaps are not an unprofiled utilization
measurement.

Run the Node tests, including the native GPU smoke check, with:

```powershell
$env:ORT_WEBGPU_NODE_MODULE = (Resolve-Path js/node).Path
node --test onnxruntime/test/webgpu/test_profile_model.mjs onnxruntime/test/webgpu/test_visualize_profile.mjs
```

Without that environment variable, the native test is explicitly skipped; the remaining tests require
no ORT build. The native test checks Abs output with CPU fallback disabled, profiling off/on/off,
multiple runs and both GPU timestamps and WebGPU CPU spans.

## Test an external Dawn host

With a compatible native Dawn host and an ORT DLL built using `--use_external_dawn`, run:

```powershell
onnxruntime_webgpu_external_dawn_test.exe --profiling
onnxruntime_webgpu_external_dawn_test.exe --profiling --host_device
```

The host test runs Abs with CPU fallback disabled. Its proc-table counters verify that timestamp
queries, writes, resolves and query buffers are absent with profiling off, present with profiling on
across two runs, and inactive again after profiling. Both ORT-created and host-created device paths
emit `first_submit` GPU events plus CPU spans. The host-created device enables `TimestampQuery`.

The test target is available when CMake has `dawn::dawn_native`. A source-Dawn build can supply the
host executable and its required Dawn/DXC libraries for an API-only ORT build: stage the executable
beside that build's `onnxruntime.dll` and verify the loaded DLL. Match the host Dawn API/ABI to the
external ORT build's Dawn pin. A host built with another Dawn version is not compatible by assumption.

Dawn also checks the proc table's version bytes. When building a Dawn source archive under an ORT Git
checkout, prevent Git discovery from picking up the parent ORT revision (for example, set the process-local
`GIT_CEILING_DIRECTORIES` to the archive's parent directory before generating its version header).
The archive host and API package should both use the no-Git version. Verify the generated API headers
and version bytes; preserve Dawn's version check rather than rewriting a proc table to bypass it.
