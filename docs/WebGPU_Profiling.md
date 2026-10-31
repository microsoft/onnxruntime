# WebGPU CPU and GPU timelines

Enable ONNX Runtime profiling as usual. For example, with the native Windows WebGPU provider:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.enable_profiling = True
options.add_session_config_entry("ep.webgpuexecutionprovider.dawnBackendType", "D3D12")
session = ort.InferenceSession("model.onnx", options, providers=["WebGpuExecutionProvider"])
session.run(None, inputs)
profile_path = session.end_profiling()
```

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

## Capture and visualize a decoder

The WebGPU test tooling includes a direct ORT decoder runner and a self-contained HTML trace viewer:

```powershell
python -m pip install numpy tokenizers plotly
python onnxruntime/test/webgpu/profile_decoder.py <model-directory> --output-dir build/profile --decode-steps 16
python onnxruntime/test/webgpu/visualize_profile.py <trace-json> --metadata build/profile/capture.json --output build/profile/timeline.html
```

Use a native WebGPU ORT Python build containing this change. An external Dawn API-only Python build
requires a host that injects a compatible Dawn proc table in the same process. The decoder directory
must contain `genai_config.json`, the ONNX model/weights and `tokenizer.json`, with the standard
input IDs, attention mask and separate past/present key/value names. The runner uses batch 1, greedy
selection and GPU-resident KV outputs, with CPU logits and graph capture disabled. It performs fixed
decode steps even after EOS; it does not apply chat templates or LoRA adapters, or reproduce a full
GenAI application's generation loop. Use `--prompt-file`, `--warmup`, `--repeat` and `--decode-steps`
to specify the workload. `--no-profile` records ordinary wall times without enabling profiling.

Open `timeline.html` locally. CPU major steps and GPU kernels share one horizontal time axis;
select a run to zoom into prefill or decode, then hover for program names and abbreviated shapes.
Full event arguments, including cache keys, remain in the original JSON trace.
The run overview shows the interval union of GPU compute and time outside those kernels. Gaps can
include CPU preparation, transfers and scheduling; transfers do not have GPU compute timestamps.
The report preserves the recorded timestamps and prominently labels approximate alignment.
It also exports a `.summary.json` with raw run timings and GPU program totals. The visualizer can
accept any WebGPU ORT trace; omit `--metadata` when using another capture tool.

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
