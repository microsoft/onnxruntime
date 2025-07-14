---
title: AMD - ROCm
description: Instructions to execute ONNX Runtime with the AMD ROCm execution provider
parent: Execution Providers
nav_order: 701
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

## Build from source

For build instructions, please see the [BUILD page](../build/eps.md#amd-rocm). Prebuild .whl files are provided below in the requirements section and are hosted on [repo.radeon.com](https://repo.radeon.com/rocm/manylinux/). Ubuntu based docker development environments are provided in the Docker Support section. New wheels and dockers are published each ROCm release.

## Requirements

Below is the matrix of supported ROCm versions corresponding to Ubuntu builds. 

As of ROCm 6.0.2 Links for prebuild Python Wheels (.whl) are linked below corresponding to python versions for the host OS based on Ubuntu support.
All links can be found on AMD's [repo.radeon manylinux page](https://repo.radeon.com/rocm/manylinux) for each corresponding to the ROCm release.

| ONNX Runtime Version | ROCm Version | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.12 |
|---|---|---|---|---|---|
|||||||
|1.21|6.4.1|| [3.9](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp39-cp39-manylinux_2_28_x86_64.whl) | [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp310-cp310-manylinux_2_28_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp312-cp312-manylinux_2_28_x86_64.whl) |
|1.21|6.4||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/onnxruntime_rocm-1.21.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/onnxruntime_rocm-1.21.0-cp312-cp312-linux_x86_64.whl)|
|1.19|6.3.1||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|
|1.19|6.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl) | [3.12](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp312-cp312-linux_x86_64.whl)|
|1.18|6.2.4||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.4/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.18|6.2.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.18|6.2| [3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp38-cp38-linux_x86_64.whl)|| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.1.3||| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.1| [3.8](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp38-cp38-linux_x86_64.whl)|| [3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.0.2|||[3.10](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0.2/onnxruntime_rocm-inference-1.17.0-cp310-cp310-linux_x86_64.whl)||
|1.17|6.0<br/>5.7|||||
|1.16|5.6<br/>5.5<br/>5.4.2|||||
|1.15|5.4.2<br/>5.4<br/>5.3.2|||||
|1.14|5.4<br/>5.3.2|||||
|1.13|5.4<br/>5.3.2|||||
|1.12|5.2.3<br/>5.2|||||

## Docker Support

For simple workloads and/or prototyping AMD creates a Docker Images based on Ubuntu using the latest ROCm release and Supported ROCm-Pytorch builds found at [ROCM Dockerhub](https://hub.docker.com/r/rocm/onnxruntime/tags).

The intent is to get users up and running with their custom workload in python and provides an environment of prebuild ROCm, Onnxruntime and MIGraphX packages required to get started without the need to build Onnxruntime.

## Configuration Options

The ROCm Execution Provider supports the following configuration options.

### device_id

The device ID.

Default value: 0

### tunable_op_enable

Set to use TunableOp.

Default value: false

### tunable_op_tuning_enable

Set the TunableOp try to do online tuning.

Default value: false

### user_compute_stream

Defines the compute stream for the inference to run on.
It implicitly sets the `has_user_compute_stream` option. It cannot be set through `UpdateROCMProviderOptions`.
This cannot be used in combination with an external allocator.

Example python usage:

```python
providers = [("ROCMExecutionProvider", {"device_id": torch.cuda.current_device(),
                                        "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_model.onnx", sess_options=sess_options, providers=providers)
```

To take advantage of user compute stream, it is recommended to
use [I/O Binding](../api/python/api_summary.html) to bind inputs and outputs to tensors in device.

### do_copy_in_default_stream

Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false, there are
race conditions and possibly better performance.

Default value: true

### gpu_mem_limit

The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The
total device memory usage may be higher.
s: max value of C++ size_t type (effectively unlimited)

_Note:_ Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)

### arena_extend_strategy

The strategy for extending the device memory arena.

 Value                | Description                                                                  
----------------------|------------------------------------------------------------------------------
 kNextPowerOfTwo (0)  | subsequent extensions extend by larger amounts (multiplied by powers of two) 
 kSameAsRequested (1) | extend by the requested amount                                               

Default value: kNextPowerOfTwo

_Note:_ Will be over-ridden by contents of `default_memory_arena_cfg` (if specified)

### gpu_external_[alloc|free|empty_cache]

gpu_external_* is used to pass external allocators.
Example python usage:

```python
from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator

provider_option_map["gpu_external_alloc"] = str(torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address())
provider_option_map["gpu_external_free"] = str(torch_gpu_allocator.gpu_caching_allocator_raw_delete_address())
provider_option_map["gpu_external_empty_cache"] = str(torch_gpu_allocator.gpu_caching_allocator_empty_cache_address())
```

Default value: 0

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

## Samples

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

providers = [
    'ROCMExecutionProvider',
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```
