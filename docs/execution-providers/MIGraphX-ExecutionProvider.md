---
title: AMD - MIGraphX
description: Instructions to execute ONNX Runtime with the AMD MIGraphX execution provider
parent: Execution Providers
nav_order: 12
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

Since ROCm 6.0.2, AMD supplies pre-built python wheels hosted on (https://repo.radeon.com/rocm/manylinux)

## Build from source
For build instructions, please see the [BUILD page](../build/eps.md#amd-migraphx). Prebuild .whl files are provided below in the requirements section and are hosted on repo.radeon.com. Ubuntu based docker development environments are provided in the Docker Support section. New whl and dockers are published each ROCm release

## Requirements

Below is the matrix of supported ROCm versions corresponding to Ubuntu builds. 

As of ROCm 6.0.2 Links for prebuild Python Wheels (.whl) are linked below corresponding to python versions for the host OS based on Ubuntu support.
All links can be found on AMD's [repo.radeon manylinux page](https://repo.radeon.com/rocm/manylinux) for each corresponding to the ROCm release

|ONNX Runtime Version|MIGraphX ROCm Release| Python 3.8 | Python 3.9 | Python 3.10 | Python 3.12 |
|---|---|---|---|---|---|
|1.21|6.4.1|| [3.9](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp39-cp39-manylinux_2_28_x86_64.whl) | [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp310-cp310-manylinux_2_28_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp312-cp312-manylinux_2_28_x86_64.whl) |
|1.21|6.4||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/onnxruntime_rocm-1.21.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/onnxruntime_rocm-1.21.0-cp312-cp312-linux_x86_64.whl)|
|1.19|6.3.1||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|
|1.19|6.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|
|1.18|6.2.4||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.4/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.18|6.2.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.18|6.2| [3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp38-cp38-linux_x86_64.whl)|| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.1.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.1| [3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp38-cp38-linux_x86_64.whl)|| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.16|6.0.2|||[3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.16|6.0.0|||||
|1.15|5.4.x|||||
|1.14|5.4|||||
|1.14|5.4|||||
|1.13|5.4|||||
|1.13|5.3.2|||||
|1.13|5.3|||||
|1.12|5.2.3|||||
|1.12|5.2|||||


## Docker Support

For simple workloads and/or prototyping AMD creates a Docker Images based on Ubuntu using the latest ROCm release and Supported ROCm-Pytorch builds found at [ROCM Dockerhub](https://hub.docker.com/r/rocm/onnxruntime/tags)

The intent is to get users up and running with their custom workload in python and provides an environment of prebuild ROCm, Onnxruntime and MIGraphX packages required to get started without the need to build Onnxruntime 


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


## Session Variables

These flags can be invoked via the Provider Options struct [link](https://github.com/ROCm/onnxruntime/blob/rocm6.4_internal_testing/include/onnxruntime/core/session/onnxruntime_c_api.h#L615) when creating an Onnxruntime Session Object and invoking the MIGraphXExecutionProvider 

Items are added as a python dictionary when invoking the MIGraphX execution provider when using python

|Session Option Flag | Values | Description|
|---|---|---|
| device_id | INT | Select the device ID specified for the session run (default will be device 0) | 
| migraphx_fp16_enable | 1 or 0 | Enable FP16 quantization mode via the MIGraphX API of the input model. |
| migraphx_int8_enable | 1 or 0 | Enable int8 static quantization mode of the input model via the MIGraphX API. Requires calibration table path vars to be set (migraphx_int8_calibration_table_name=valid path).|
| migraphx_int8_calibration_table_name | <absolute path to calibration table> | Path to a set of input calibration data for int8 static model quantization. |
| migraphx_int8_use_native_calibration_table | 1 or 0 | Use a calibration table from Nvidia native int8 format or json dumped format. |
| migraphx_save_compiled_model | 1 or 0 | Enable saving a model as an MIGraphX (.mxr) format after compile when set to 1 |
| migraphx_save_compiled_path  | path string | Path to write .mxr file (default is ./compiled_model.mxr). Path where the MIGraphX compiled model is stored. Requires migraphx_save_compiled_path to be set for this session |
| migraphx_load_compiled_model | 1 or 0 | Enable loading a model as an MIGraphX (.mxr) format after compile when set to 1 |
| migraphx_load_compiled_path | path string | Path to read .mxr file (default is ./compiled_model.mxr). Path where the MIGraphX compiled model is stored. |
| migraphx_exhaustive_tune | 1 or 0 (default 0) | Enable exhaustive tuning of parameters as part of compilation via the MIGraphX API. Adds additional compile time for a potential perf boost.|
| migraphx_mem_limit | INT | Set the memory limit used for memory arena. Default uses ORTs default_memory_arena_cfg value. |
| migraphx_external_alloc | Address | Address of external memory allocator function used for this EP. Useful for reading in larger models weights. |
| migraphx_external_free | Address  | Address of external memory deallocator function used for this EP. Useful for unloadng what was allocated with the migraphx_external_alloc input. |
| migraphx_external_empty_cache | Address  | Address of external memory cache used for this model. Useful for caching results of externally allocated models. |


## Environment Variables

Environment variables can be invoked on a global level. These are typically used via:

```
export ORT_MIGRAPHX_XXXXX = <value> 
```

or invoked before running a command as:

```
ORT_MIGRAPHX_XXXX=<value> python3 example_script.py
```

This will start an inference session and supersede inputs invoked via 'Session()'. 

Users can invoke Environment and Session variables in the same run but Environment variables will take precident. 

|Environment Option Flag | Values | Description|
|---|---|---|
| ORT_MIGRAPHX_DUMP_MODEL_OPS | 1 or 0 | Enable dumping of model operators during parsing. | 
| ORT_MIGRAPHX_FP16_ENABLE | 1 or 0 | Enable FP16 quantization mode via the MIGraphX API of the input model. |
| ORT_MIGRAPHX_INT8_ENABLE | 1 or 0 | Enable int8 static quantization mode of the input model via the MIGraphX API.\n Requires calibration table path vars to be set (migraphx_int8_calibration_table_name=<valid path>).|
| ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME | <absolute path to calibration table> | Path to a set of input calibration data for int8 static model quantization. |
| ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE | 1 or 0 | Use a calibration table from Nvidia native int8 format or json dumped format. |
| ORT_MIGRAPHX_SAVE_COMPILED_MODEL | 1 or 0 | Enable saving a model as an MIGraphX (.mxr) format after compile. |
| ORT_MIGRAPHX_SAVE_COMPILED_PATH | <string> - Path to write .mxr file (default is ./compiled_model.mxr) | Path where the MIGraphX compiled model is stored. Requires ORT_MIGRAPHX_SAVE_COMPILED_MODEL to be set to 1. |
| ORT_MIGRAPHX_LOAD_COMPILED_MODEL | 1 or 0 | Enable loading a model as an MIGraphX (.mxr) format after compile. |
| ORT_MIGRAPHX_LOAD_COMPILED_PATH | <string> - Path to read .mxr file (default is ./compiled_model.mxr) | Path where the MIGraphX compiled model is stored. Requires ORT_MIGRAPHX_LOAD_COMPILED_MODEL to be set to 1. |
| ORT_MIGRAPHX_EXHAUSTIVE_TUNE | 1 or 0 (default 0) | Enable exhaustive tuning of parameters as part of compilation via the MIGraphX API. Adds additional compile time for a potential perf boost. |



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
