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

Since ROCm 6.0.2, AMD supplies pre-built python wheels hosted on (https://repo.radeon.com/manylinux)

## Requirements

Below is the matrix of supported ROCm versions corresponding to Ubuntu builds.

Links for prebuild Python Wheels (.whl) are linked below corresponding to python versions for the host OS based on Ubuntu support.


|ONNX Runtime Version|MIGraphX ROCm Release| Repo.Radeon link|
|---|---|---|
|main|5.4||
|1.14|5.4||
|1.13|5.4||
|1.13|5.3.2||
|1.12|5.2.3||
|1.12|5.2||
|1.13|5.3||
|1.14|5.4||
|1.15|5.4.x||
|1.16|6.0.0||
|1.16|6.0.2| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)|
|1.17|6.1| [Python 3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp38-cp38-linux_x86_64.whl) [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)|
|1.17|6.1.3| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)|
|1.18|6.2| [Python 3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp38-cp38-linux_x86_64.whl) [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)|
|1.18|6.2.3| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)|
|1.18|6.2.4| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.4/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)|
|1.19|6.3| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) [Python 3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|
|1.19|6.3.1| [Python 3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) [Python 3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|

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
