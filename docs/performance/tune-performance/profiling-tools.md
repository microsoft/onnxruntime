---
title: Profiling tools 
grand_parent: Performance
parent: Tune performance
nav_order: 1
---

# Profiling Tools

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## In-code performance profiling

The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`. The [perf_view tool](https://github.com/microsoft/onnxruntime/tree/main/tools/perf_view) can also be used to render the statistics as a summarized view in the browser.

You can enable ONNX Runtime latency profiling in code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```

If you are using the onnxruntime_perf_test.exe tool, you can add `-p [profile_file]` to enable performance profiling.

In both cases, you will get a JSON file which contains the detailed performance data (threading, latency of each operator, etc). This file is a standard performance tracing file, and to view it in a user-friendly way, you can open it by using multiple tools.

* (Windows) Use the WPA GUI to open the trace using the Perfetto OSS plugin - [Microsoft-Performance-Tools-Linux-Android](https://github.com/microsoft/Microsoft-Performance-Tools-Linux-Android)
* [Perfetto UI](https://www.ui.perfetto.dev/) - Successor to Chrome Tracing UI
* chrome://tracing: 
  * Open a Chromium based browser such as Edge or Chrome
  * Type chrome://tracing in the address bar
  * Load the generated JSON file

## Execution Provider (EP) Profiling

Starting with ONNX 1.17 support has been added to profile EPs or Neural Processing Unit (NPU)s, if that EP supports profiling in it's SDK

## Qualcomm QNN EP

As mentioned in the [QNN EP Doc](../../execution-providers/QNN-ExecutionProvider.md) profiling is supported

### Cross-Platform CSV Tracing

The Qualcomm AI Engine Direct SDK (QNN SDK) supports profiling. QNN will output to CSV in a text format if a dev were to use the QNN SDK directly outside ONNX. To enable equivalent functionality, ONNX mimics this support and outputs the same CSV formatting.

If profiling_level is provided then ONNX will append log to current working directory a qnn-profiling-data.csv [file](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/qnn/builder/qnn_backend_manager.cc#L911)

### TraceLogging ETW (Windows) Profiling

As covered in [logging](logging_tracing.md) ONNX supports dynamic enablement of tracing ETW providers. Specifically the following settings. If the Tracelogging provider is enabled and profiling_level was provided, then CSV support is automatically disabled

- Provider Name: Microsoft.ML.ONNXRuntime  
- Provider GUID: 3a26b1ff-7484-7484-7484-15261f42614d  
- Keywords: Profiling = 0x100  per [logging.h](https://github.com/ivberg/onnxruntime/blob/user/ivberg/ETWRundown/include/onnxruntime/core/common/logging/logging.h#L81)  
- Level: 
  - 5 (VERBOSE) = profiling_level=basic (good details without perf loss)
  - greater than 5 = profiling_level=detailed (individual ops are logged with inference perf hit)  
- Event: [QNNProfilingEvent](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/qnn/builder/qnn_backend_manager.cc#L1083)

## GPU Profiling

To profile CUDA kernels, please add the cupti library to your PATH and use the onnxruntime binary built from source with `--enable_cuda_profiling`.
To profile ROCm kernels, please add the roctracer library to your PATH and use the onnxruntime binary built from source with `--enable_rocm_profiling`. 

Performance numbers from the device will then be attached to those from the host. For example:

```json
{"cat":"Node", "name":"Add_1234", "dur":17, ...}
{"cat":"Kernel", "name":"ort_add_cuda_kernel", dur:33, ...}
```

Here, the "Add" operator from the host initiated a CUDA kernel on device named "ort_add_cuda_kernel" which lasted for 33 microseconds.
If an operator called multiple kernels during execution, the performance numbers of those kernels will all be listed following the call sequence:

```json
{"cat":"Node", "name":<name of the node>, ...}
{"cat":"Kernel", "name":<name of the kernel called first>, ...}
{"cat":"Kernel", "name":<name of the kernel called next>, ...}
```
