---
title: MIGraphX (AMD)
description: Instructions to execute ONNX Runtime with the AMD MIGraphX execution provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 4
redirect_from: /docs/reference/execution-providers/MIGraphX-ExecutionProvider
---

# MIGraphX Execution Provider
{: .no_toc }

The [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/) execution provider uses AMD's Deep Learning graph optimization engine to accelerate ONNX model on AMD GPUs. 

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build
For build instructions, please see the [BUILD page](../build/eps.md#amd-migraphx). 

## Usage

### C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MiGraphX(sf, device_id));
```

You can check [here](https://github.com/scxiao/ort_test/tree/master/char_rnn) for a specific c/c++ program.

The C API details are [here](../get-started/with-c.md).

### Python
When using the Python wheel from the ONNX Runtime build with MIGraphX execution provider, it will be automatically
prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution
provider. Python APIs details are [here](https://onnxruntime.ai/docs/api/python/api_summary.html).
*Note that the next release (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution provider other than the default CPU provider when instantiating InferenceSession.*

You can check [here](https://github.com/scxiao/ort_test/tree/master/python/run_onnx) for a python script to run an
model on either the CPU or MIGraphX Execution Provider.

## Configuration Options
MIGraphX providers an environment variable ORT_MIGRAPHX_FP16_ENABLE to enable the FP16 mode.

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../performance/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e migraphx`
