---
title: AMD - ROCm
description: Instructions to execute ONNX Runtime with the AMD ROCm execution provider
parent: Execution Providers
nav_order: 10
redirect_from: /docs/reference/execution-providers/ROCm-ExecutionProvider
---

# ROCm Execution Provider
{: .no_toc }

The ROCm Execution Provider enables hardware accelerated computation on AMD ROCm-enabled GPUs. 

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

**NOTE** Please make sure to install the proper version of Pytorch specified here [PyTorch Version](../install/#training-install-table-for-all-languages).

For Nightly PyTorch builds please see [Pytorch home](https://pytorch.org/) and select ROCm as the Compute Platform.

Pre-built binaries of ONNX Runtime with ROCm EP are published for most language bindings. Please reference [Install ORT](../install).

## Requirements


|ONNX Runtime|ROCm|
|---|---|
|main|5.4|
|1.13|5.4|
|1.13|5.3.2|
|1.12|5.2.3|
|1.12|5.2|


## Build
For build instructions, please see the [BUILD page](../build/eps.md#amd-rocm). 

## Usage

### C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCm(so, device_id));
```

The C API details are [here](../get-started/with-c.md).

### Python
Python APIs details are [here](https://onnxruntime.ai/docs/api/python/api_summary.html).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../performance/tune-performance.md)

## Samples

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

providers = [
    'ROCmExecutionProvider',
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```
