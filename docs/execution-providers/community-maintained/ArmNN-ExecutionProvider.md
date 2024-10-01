---
title: Arm - Arm NN
description: Instructions to execute ONNX Runtime with Arm NN on Armv8 cores
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 2
redirect_from: /docs/reference/execution-providers/ArmNN-ExecutionProvider
---

# Arm NN Execution Provider
{: .no_toc}

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

Accelerate performance of ONNX model workloads across Arm®-based devices with the Arm NN execution provider. [Arm NN](https://github.com/ARM-software/armnn) is an open source inference engine maintained by Arm and Linaro companies. 

## Build
For build instructions, please see the [BUILD page](../../build/eps.md#arm-nn).

## Usage
### C/C++
To use Arm NN as execution provider for inferencing, please register it as below.
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
bool enable_cpu_mem_arena = true;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(so, enable_cpu_mem_arena));
```
The C API details are [here](../../get-started/with-c.md).

## Performance Tuning
When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest), use the flag -e armnn
