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

With a compatible native Dawn host and an ORT DLL built using `--use_external_dawn`, run:

```powershell
onnxruntime_webgpu_external_dawn_test.exe --profiling
onnxruntime_webgpu_external_dawn_test.exe --profiling --host_device
```

The host test runs Abs with CPU fallback disabled. Its proc-table counters verify that timestamp
queries, writes, resolves and query buffers are absent with profiling off, present with profiling on
across two runs, and inactive again after profiling. Both ORT-created and host-created device paths
emit `first_submit` GPU events plus CPU spans. The host-created device enables `TimestampQuery`.

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
