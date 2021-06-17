---
title: ARM NN
parent: Execution Providers
grand_parent: Reference
---

# ArmNN Execution Provider
{: .no_toc}

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

[ArmNN](https://github.com/ARM-software/armnn) is an open source inference engine maintained by Arm and Linaro companies. The integration of ArmNN as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across Armv8 cores.

## Build
For build instructions, please see the [BUILD page](../../how-to/build/eps.md#armnn).

## Usage
### C/C++
To use ArmNN as execution provider for inferencing, please register it as below.
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
bool enable_cpu_mem_arena = true;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(sf, enable_cpu_mem_arena));
```
The C API details are [here](../api/c-api.md).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../../how-to/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest), use the flag -e armnn
