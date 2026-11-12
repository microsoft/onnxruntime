# WebGPU CPU and GPU timelines

Enable ONNX Runtime profiling in the application that runs your workload, including applications
using ONNX Runtime GenAI. Save the profiling JSON, then process it offline with Node.js to generate
an interactive webpage. The reporting script does not load models or run inference.

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

External Dawn and custom devices/proc tables can provide the host callback described below to obtain
the same Windows D3D12 correlation. The submission-time approximation remains in use without a host
callback, on other backends/platforms, or when calibration fails or uses incompatible timestamp toggles.
GPU events require a device with timestamp-query support. Concurrent profiling of sessions sharing
a WebGPU context is unsupported.

WebGPU also records CPU-side `WebGPU::Encode`, `WebGPU::Finish`, `WebGPU::Submit`, `WebGPU::Wait`,
`WebGPU::StartProfiling` and `WebGPU::ProfileReadback` spans on the same CPU time origin.
These supplement the existing ORT model-run, executor and operator events without changing the core
profiler. They identify command preparation, submission, waits and profiling overhead. Nested spans
overlap: for example, a pipeline wait can be inside command encoding. Do not sum them as independent work.

Each profiled `WebGPU::Submit` and its GPU dispatches carry the same `submission_id` in their event
arguments. IDs distinguish submissions across runs and recreated contexts in a process. **CPU → GPU links**
is enabled by default: blue arrows connect the end of `queue.submit` to the first GPU dispatch in each batch.
Dense views show at most 80 overview arrows and state the displayed count; zoom in to see more.
Select a CPU submission or GPU dispatch to highlight the batch and draw arrows to all its GPU dispatches.
The event inspector lists related events; **Zoom to submission** shows the CPU call and its GPU work,
including links outside the selected inference interval. Older logs remain readable without links.

Use **Fit GPU work** to zoom to the GPU dispatches in the selected interval. At wider scales,
events narrower than three pixels appear as markers; the report identifies this minimum width,
and the inspector retains the measured duration. Small nonzero times are not rounded to zero.

`WebGPU::Submit` measures the `queue.submit` call; command-buffer finishing is recorded separately as
`WebGPU::Finish` on the encoding lane. Arrows start at the recorded end of the CPU submit call and
end at the GPU dispatch's recorded start. GPU execution can begin before the CPU call returns:
these links point backwards in time. With calibrated clocks, the inspector shows the signed
submit-end-to-first-GPU-start interval. Negative values indicate CPU/GPU overlap; positive values
are the gap after the submit call returns and can include queueing and scheduling. This interval
does not measure total GPU queueing time. Approximate `first_submit` timestamps retain the recorded
dependency but cannot measure that interval. IDs and links add no GPU operations
or synchronization, and their runtime bookkeeping is active only during profiling.

Clock calibration, CPU event clocks/storage, timestamp writes and query resolve/readback run only
while profiling is active. Timestamp feature negotiation at device creation remains unchanged, so a
session can enable run-level profiling later. Profiling adds measurement and synchronization overhead;
profiled wall times should not be treated as normal inference throughput.
CPU timing gates are inline: disabled execution skips clock reads and event-recording calls and
allocates no event data.

## Enable profiling in the application

Use the application's existing generation loop, inputs, provider configuration and graph-capture
settings. Its ORT runtime must contain this WebGPU profiling change. Console/verbose logging is
separate from profiling: the reporting tool consumes ORT profiling JSON files.

For ONNX Runtime GenAI, add a profiling file prefix to the existing
`model.decoder.session_options` object in `genai_config.json`. This fragment shows the field to merge;
keep the rest of the model configuration:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "enable_profiling": "webgpu_profile"
      }
    }
  }
}
```

Run the application normally, then release the generator/model sessions cleanly so ORT finishes
writing the profile. Session-level profiling retains the intervals between inference runs and
produces a file named like `webgpu_profile_<timestamp>.json` per profiled session.
For a direct ORT C++ application, use the existing `Ort::SessionOptions::EnableProfiling` API before
session creation and `Ort::Session::EndProfilingAllocated` or session destruction to finish the file.

GenAI versions with the `enable_profiling` runtime option also support enabling profiling for selected
generation steps. In C++, use the existing generator's `SetRuntimeOption("enable_profiling", "1")`
to enable it, or supply a custom file prefix instead of `"1"`; use `"0"` to disable it. This requires
ORT 1.25 or newer and produces a separate profile for each subsequent inference run. See
[GenAI runtime options](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/RuntimeOptions.md).
Use session-level profiling when you need to inspect gaps between generation steps.

## Process the saved log with Node.js

The dependency-free reporting script requires Node.js 20 or newer. It needs only the saved JSON
trace and the checked-in report templates; no model, tokenizer, input manifest, Python, GenAI or
`onnxruntime-node` installation is needed for post-processing:

```powershell
node onnxruntime/test/webgpu/profiling/visualize_profile.mjs webgpu_profile_<timestamp>.json --output build/profile/timeline.html
```

This writes self-contained `timeline.html` and `timeline.summary.json`, preserving the original
trace. Open the HTML in a current Chromium browser. If local-file viewing is restricted, serve the
output with a local HTTP server. The webpage makes no network requests.

The default **Whole trace** view aligns all recorded CPU major steps and GPU dispatches on one axis,
including gaps between inference runs and events outside individual `model_run` intervals. Select a
run to inspect it, search for an operator/kernel, click events for timings/shapes, drag to zoom,
Shift-drag to pan, or export the timeline PNG/summary JSON. Whole trace always includes all recorded
runs; stage/warmup filters affect the run list and comparison.

Both ORT event arrays and objects containing `traceEvents` are accepted. GPU events must contain the
WebGPU `clock_alignment` field described above. Process each session-level or run-level file separately:
their profiler time origins can differ, so concatenating files would not preserve alignment.

No additional metadata is required. Optional `--metadata annotations.json` can label runs with
application-known stages or warmup flags; its `runs` array must match the trace's `model_run` count
and order. The tool does not guess prefill/decode boundaries or warmup from kernel names. For example:

```json
{
  "runs": [
    { "stage": "prefill", "warmup": false },
    { "stage": "decode", "warmup": false }
  ]
}
```

An optional `--baseline baseline.json` displays separate unprofiled application measurements. Both
metadata files must identify the same `model_sha256`, `ort_version`, `backend` and `workload`;
baseline `trace` must be null and its `runs` supply `stage`, `warmup` and `wall_ms`. Optional input
hashes/session options are compared when present. Model weights, workload equivalence and machine
conditions must be verified by the application.

ORT profiling records ORT execution and WebGPU CPU work. GenAI work outside ORT, such as tokenization,
sampling and application scheduling, has no named CPU span here; it can appear as a gap between runs.
GPU timestamps cover compute dispatches, not every transfer or other GPU workload. Profiling adds
query, trace-collection and synchronization overhead, so time outside kernels is not an unprofiled
GPU-utilization measurement. Measure throughput separately with profiling off.

Run the Node post-processing tests without an ORT build:

```powershell
node --test onnxruntime/test/webgpu/profiling/test_visualize_profile.mjs
```

## Test an external Dawn host

External ORT builds do not link Dawn's native backend. Standard WebGPU procs do not expose its D3D12
queue, so native access must happen in the application using the Dawn implementation that owns the
device. Include `core/providers/webgpu/webgpu_provider_factory.h` and supply the optional version 1
`OrtWebGpuClockCalibrationCallbackInfo` through the existing WebGPU provider options:

```cpp
int ORT_API_CALL Calibrate(void* device, OrtWebGpuClockCalibration* result, void* user_data) noexcept;

OrtWebGpuClockCalibrationCallbackInfo calibration{1, Calibrate, &host_state};
provider_options["clockCalibration"] = std::to_string(reinterpret_cast<uintptr_t>(&calibration));
session_options.AppendExecutionProvider("WebGPU", provider_options);
```

ORT copies the callback struct during provider creation. The callback and `user_data` must outlive
the WebGPU context; all sessions sharing that context must specify the same callback and user data.
The address must point to a valid struct in the current process, as with `dawnProcTable`. ORT rejects
unsupported versions, null callbacks and conflicting shared-context configurations.

The callback receives the active opaque `WGPUDevice` and returns nonzero after writing all three fields:

- `gpu_timestamp_ns`: the calibration GPU timestamp in exactly the same units as resolved queries.
- `cpu_qpc_ticks`: the correlated Windows QueryPerformanceCounter ticks, not CPU nanoseconds.
- `gpu_to_cpu_scale`: a finite positive factor converting resolved GPU query deltas to CPU nanoseconds.

Use the owning Dawn implementation to obtain the D3D12 queue and call `GetClockCalibration`. Match
that Dawn version's query normalization and timestamp toggles, including any rate correction; using
the exact GPU frequency for an absolute timestamp can disagree with Dawn's rounded conversion.
The test host's `CalibrateHostDevice` in
[`external_dawn/main.cc`](../onnxruntime/test/webgpu/external_dawn/main.cc) is a working example for
the pinned Dawn version. It rejects `disable_timestamp_query_conversion` and `timestamp_quantization`.
Recheck the conversion when updating Dawn. The callback must not throw, wait for GPU work, or submit
extra commands. Return zero when calibration is unavailable.

ORT calls the callback only during Windows D3D12 profiling, once per inference run, and bridges QPC
to the existing profiler origin. Failed callbacks or invalid samples use `first_submit`; successful
callbacks produce `d3d12_clock_calibration`. No callback or clock sampling occurs with profiling off.

With a compatible native Dawn host and an ORT DLL built using `--use_external_dawn`, run:

```powershell
onnxruntime_webgpu_external_dawn_test.exe --profiling
onnxruntime_webgpu_external_dawn_test.exe --profiling --host_device
onnxruntime_webgpu_external_dawn_test.exe --profiling --clock_calibration
onnxruntime_webgpu_external_dawn_test.exe --profiling --clock_calibration --host_device
```

The host test runs Abs with CPU fallback disabled. Its proc-table counters verify that timestamp
queries, writes, resolves and query buffers are absent with profiling off, present with profiling on
across two runs, and inactive again after profiling. Both ORT-created and host-created device paths
emit CPU spans and GPU events. Without the callback they use `first_submit`. With `--clock_calibration`
on Windows D3D12 they use `d3d12_clock_calibration`, with zero callback calls off and two calls on.
The host-created device enables `TimestampQuery` and disables `timestamp_quantization` for calibration
tests. Optional `--calibration_failure` and
`--calibration_invalid` flags exercise fallback for failed callbacks and invalid samples.

Run the complete Windows D3D12 regression matrix with Node.js after building the host:

```powershell
node onnxruntime/test/webgpu/profiling/test_external_dawn_profile.mjs build/external-regression/Release/Release/onnxruntime_webgpu_external_dawn_test.exe build/profile/external-tests
```

The script checks both device paths with absent, successful, failed and invalid calibration, plus
missing-proc rejection. It saves logs and traces, verifies all six CPU span kinds and submission links, and checks that
the calibrated GPU events preserve an intentionally delayed queue. The delay is test-only; production
calibration adds no queue wait. Generate a webpage from any saved `.trace.json` with `visualize_profile.mjs`.

Build the test in external-Dawn mode with Dawn sources available for the native host dependency:

```powershell
$env:GIT_CEILING_DIRECTORIES = "$PWD/build/external-regression/Release/_deps"
.\build.bat --config Release --build_dir build/external-regression --use_webgpu --use_external_dawn --build_shared_lib --update --build --parallel 12 --skip_tests --target onnxruntime_webgpu_external_dawn_test --cmake_extra_defines onnxruntime_ENABLE_DAWN_BACKEND_D3D12=ON onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=OFF
```

Here Dawn's source dependencies supply the native test host; `--use_external_dawn` keeps the native
Dawn backend out of the ORT DLL. The test executable links native Dawn and supplies its proc table.
Do not set `onnxruntime_DAWN_PREBUILT_DIR` in this test build: the API-only package does not provide
`dawn::dawn_native`, which the test requires. Run both commands above from the build output directory.

To test a separate API-only ORT build, stage the host executable and its required Dawn/DXC libraries
beside that build's `onnxruntime.dll` and verify the loaded DLL. Match the host Dawn API/ABI to the
external ORT build's Dawn pin. A host built with another Dawn version is not compatible by assumption.

Dawn also checks the proc table's version bytes. When building a Dawn source archive under an ORT Git
checkout, prevent Git discovery from picking up the parent ORT revision (for example, set the process-local
`GIT_CEILING_DIRECTORIES` to the archive's parent directory before generating its version header).
The archive host and API package should both use the no-Git version. Verify the generated API headers
and version bytes; preserve Dawn's version check rather than rewriting a proc table to bypass it.
