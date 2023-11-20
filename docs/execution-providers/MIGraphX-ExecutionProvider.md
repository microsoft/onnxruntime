---
title: AMD - MIGraphX
description: Instructions to execute ONNX Runtime with the AMD MIGraphX execution provider
parent: Execution Providers
nav_order: 11
redirect_from: /docs/reference/execution-providers/MIGraphX-ExecutionProvider
---

# MIGraphX Execution Provider
{: .no_toc }

The [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/) execution provider uses AMD's Deep Learning graph optimization engine to accelerate ONNX model on AMD GPUs.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}
## Install

**NOTE** Please make sure to install the proper version of Pytorch specified here [PyTorch Version](../install/#training-install-table-for-all-languages).

For Nightly PyTorch builds please see [Pytorch home](https://pytorch.org/) and select ROCm as the Compute Platform.

Pre-built binaries of ONNX Runtime with MIGraphX EP are published for most language bindings. Please reference [Install ORT](../install).

## Requirements


|ONNX Runtime|MIGraphX|
|---|---|
|main|5.4|
|1.14|5.4|
|1.13|5.4|
|1.13|5.3.2|
|1.12|5.2.3|
|1.12|5.2|


## Build
For build instructions, please see the [BUILD page](../build/eps.md#amd-migraphx).

## Usage

### C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(so, device_id));
```

The C API details are [here](../get-started/with-c.md).

### Python

When using the Python wheel from the ONNX Runtime build with MIGraphX execution provider, it will be automatically
prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution
provider.

Python APIs details are [here](https://onnxruntime.ai/docs/api/python/api_summary.html).

*Note that the next release (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution provider other than the default CPU provider when instantiating InferenceSession.*

You can check [here](https://github.com/scxiao/ort_test/tree/master/python/run_onnx) for a python script to run an
model on either the CPU or MIGraphX Execution Provider.


## Configuration Options
MIGraphX providers an environment variable ORT_MIGRAPHX_FP16_ENABLE to enable the FP16 mode.

## Samples

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

providers = [
    'MIGraphXExecutionProvider',
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```
