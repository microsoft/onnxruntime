---
title: AMD MI GraphX
parent: Execution Providers
grand_parent: Reference
nav_order: 7
---

# MIGraphX Execution Provider
{: .no_toc }

The [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/) execution provider uses AMD's Deep Learning graph optimization engine to accelerate ONNX model on AMD GPUs. 

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build
For build instructions, please see the [BUILD page](../../how-to/build/eps.md#amd-migraphx). 

## Usage

### C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MiGraphX(sf, device_id));
```

You can check [here](https://github.com/scxiao/ort_test/tree/master/char_rnn) for a specific c/c++ program.

The C API details are [here](../api/c-api.md).

### Python
When using the Python wheel from the ONNX Runtime build with MIGraphX execution provider, it will be automatically
prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution
provider. Python APIs details are [here](/python/api_summary).

You can check [here](https://github.com/scxiao/ort_test/tree/master/python/run_onnx) for a python script to run an
model on either the CPU or MIGraphX Execution Provider.

## Configuration Options
MIGraphX providers an environment variable ORT_MIGRAPHX_FP16_ENABLE to enable the FP16 mode.

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../../how-to/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e migraphx` 