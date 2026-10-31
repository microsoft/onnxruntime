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
