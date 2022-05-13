---
title: Performance Tuning Tools
parent: Performance
nav_order: 2
description: Tools for tuning ONNX Runtime inference models across different Execution Providers and programming languages including OLive tool.

redirect_from: /docs/how-to/tune-performance
---
<div class="container">


# Performance Tuning Tools

Here are the tools for tuning your ONNX Runtime inference models across different Execution Providers and programming languages.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## ONNX Go Live (OLive) Tool

The [ONNX Go Live (OLive) tool](https://github.com/microsoft/OLive) is a Python package that automates the process of accelerating models with ONNX Runtime (ORT).

It contains two parts:

1. model conversion to ONNX with correctness checking
2. auto performance tuning with ORT

Users can run these two together through a single pipeline or run them independently as needed.

As a quick start to using the Microsoft ONNX OLive tool, please refer to the [notebook tutorials](https://github.com/microsoft/OLive/tree/master/notebook-tutorial) and [command line examples.](https://github.com/microsoft/OLive/tree/master/cmd-example)


## onnxruntime_perf_test.exe tool

The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. Please find the usage instructions using `onnxruntime_perf_test.exe -h`.

You can enable ONNX Runtime latency profiling in the Python code:

```python
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
```

If you are using the onnxruntime_perf_test.exe tool, you can add `-p [profile_file]` to enable performance profiling.

### Performance and Profiling Report

In both the cases, you will get a JSON file which contains the detailed performance data (threading, latency of each operator, and so on). This file is a standard performance tracing file, and to view it in a user-friendly way, you can open it by using chrome://tracing:

1. Open Chrome browser
2. Type chrome://tracing in the address bar
3. Load the generated JSON file

### Profiling CUDA Kernels

To profile Compute Unified Device Architecture (CUDA) kernels, please add 'cupti' library to PATH and use onnxruntime binary built from source with `--enable_cuda_profiling`. The performance numbers from device will then be attached to those from the host.

**Single Kernel example:**

In the following example, the "Add" operator from the host initiated a CUDA kernel on device named "ort_add_cuda_kernel" which lasted for 33 microseconds.

- JSON:

```json
{"cat":"Node", "name":"Add_1234", "dur":17, ...}
{"cat":"Kernel", "name":"ort_add_cuda_kernel", "dur":33, ...}
```

**Multiple Kernels example:**

If an operator called multiple kernels during execution, the performance numbers of all those kernels will be listed following the calling sequence:

- JSON:

```json
{"cat":"Node", "name":<name of the node>, ...}
{"cat":"Kernel", "name":<name of the kernel called first>, ...}
{"cat":"Kernel", "name":<name of the kernel called next>, ...}
```
## Ort_perf_view tool

ONNX Runtime also offers a [tool](https://github.com/microsoft/onnxruntime/tree/master/tools/perf_view) to render the statistics as a summarized view in the browser.

The tool takes the input as a JSON file and reports the performance of the GPU and CPU in the form of a treemap in the browser.

<p><a href="#" id="back-to-top">Back to top</a></p>


</div>