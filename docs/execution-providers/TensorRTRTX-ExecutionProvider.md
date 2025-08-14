---
title: NVIDIA - TensorRT RTX
description: Instructions to execute ONNX Runtime on NVIDIA RTX GPUs with the Nvidia TensorRT RTX execution provider
parent: Execution Providers
nav_order: 17
redirect_from: /docs/reference/execution-providers/TensorRTRTX-ExecutionProvider
---

# Nvidia TensorRT RTX Execution Provider
{: .no_toc }

Nvidia TensorRT RTX execution provider is the preferred execution provider for GPU acceleration on consumer hardware (RTX PCs). It is more straightforward to use than the datacenter focused legacy TensorRT Execution provider and more performant than CUDA EP. 
Just some of the things that make it a better fit on RTX PCs than our legacy TensorRT Execution Provider: 
* Much smaller footprint
* Much faster model compile/load times.
* Better usability in terms of use of cached models across multiple RTX GPUs.

The Nvidia TensorRT RTX execution provider in the ONNX Runtime makes use of NVIDIA's [TensorRT RTX](https://developer.nvidia.com/tensorrt-rtx) Deep Learning inferencing engine to accelerate ONNX models on RTX GPUs. Microsoft and NVIDIA worked closely to integrate the TensorRT RTX execution provider with ONNX Runtime.

Currently TensorRT RTX supports RTX GPUs from Ampere or later architectures. Support for Turing GPUs is coming soon.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install
Please select the Nvidia TensorRT RTX version of Onnx Runtime: https://onnxruntime.ai/docs/install. (TODO!)

## Build from source
See [Build instructions](../build/eps.md#tensorrt-rtx).

## Requirements

| ONNX Runtime | TensorRT-RTX | CUDA           |
| :----------- | :----------- | :------------- |
| main         | 1.0          | 12.0-12.9      |
| 1.22         | 1.0          | 12.0-12.9      |

## Usage
### C/C++
```c++
const auto& api = Ort::GetApi();
Ort::SessionOptions session_options;
api.SessionOptionsAppendExecutionProvider(session_options, "NvTensorRtRtx", nullptr, nullptr, 0);
Ort::Session session(env, model_path, session_options);
```

The C API details are [here](../get-started/with-c.md).

### Python
To use TensorRT RTX execution provider, you must explicitly register TensorRT RTX execution provider when instantiating the `InferenceSession`.

```python
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx', providers=['NvTensorRtRtxExecutionProvider'])
```

## Configurations
TensorRT RTX settings can be configured via [TensorRT Execution Provider Session Option](./TensorRTRTX-ExecutionProvider.md#execution-provider-options).

Here are examples and different [scenarios](./TensorRTRTX-ExecutionProvider.md#scenario) to set NV TensorRT RTX EP session options:

#### Click below for Python API example:

<details>

```python
import onnxruntime as ort

model_path = '<path to model>'

# note: for bool type options in python API, set them as False/True
provider_options = {
  'device_id': 0,
  'nv_dump_subgraphs': False,
  'nv_detailed_build_log': True,
  'user_compute_stream': stream_handle
}

sess_opt = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=[('NvTensorRTRTXExecutionProvider', provider_options)])
```

</details>

#### Click below for C++ API example:

<details>

```c++
Ort::SessionOptions session_options;

cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);

// Need to put the CUDA stream handle in a string
char streamHandle[32];
sprintf_s(streamHandle, "%lld", (uint64_t)cuda_stream);

const auto& api = Ort::GetApi();
std::vector<const char*> option_keys = {
    "device_id",
    "user_compute_stream",  // this implicitly sets "has_user_compute_stream"
};
std::vector<const char*> option_values = {
    "1",
    streamHandle
};

Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider(session_options, "NvTensorRtRtx", option_keys.data(), option_values.data(), option_keys.size()));

```

</details>

### Scenario

| Scenario                                           | NV TensorRT RTX EP Session Option                                                          | Type   |
| :------------------------------------------------- | :----------------------------------------------------------------------------------------- | :----- |
| Specify GPU id for execution                       | [device_id](./TensorRTRTX-ExecutionProvider.md#device_id)                                  | int    |
| Set custom compute stream for GPU operations       | [user_compute_stream](./TensorRTRTX-ExecutionProvider.md#user_compute_stream)              | string |
| Set TensorRT RTX EP GPU memory usage limit         | [nv_max_workspace_size](./TensorRTRTX-ExecutionProvider.md#nv_max_workspace_size)          | int    |
| Dump optimized subgraphs for debugging             | [nv_dump_subgraphs](./TensorRTRTX-ExecutionProvider.md#nv_dump_subgraphs)                  | bool   |
| Capture CUDA graph for reduced launch overhead     | [nv_cuda_graph_enable](./TensorRTRTX-ExecutionProvider.md#nv_cuda_graph_enable)            | bool   |
| Enable detailed logging of build steps             | [nv_detailed_build_log](./TensorRTRTX-ExecutionProvider.md#nv_detailed_build_log)          | bool   |
| Define min shapes                                  | [nv_profile_min_shapes](./TensorRTRTX-ExecutionProvider.md#nv_profile_min_shapes)          | string |
| Define max shapes                                  | [nv_profile_max_shapes](./TensorRTRTX-ExecutionProvider.md#nv_profile_max_shapes)          | string |
| Define optimal shapes                              | [nv_profile_opt_shapes](./TensorRTRTX-ExecutionProvider.md#nv_profile_opt_shapes)          | string |

> Note: for bool type options, assign them with **True**/**False** in python, or **1**/**0** in C++.

### Execution Provider Options

TensorRT RTX configurations can be set by execution provider options. It's useful when each model and inference session have their own configurations. All configurations should be set explicitly, otherwise default value will be taken.

##### device_id 

* Description: GPU device ID.
* Default value: 0

##### user_compute_stream

* Description: define the compute stream for the inference to run on. It implicitly sets the `has_user_compute_stream` option. The stream handle needs to be printed on a string as decimal number and passed down to the session options as shown in the example above.

* This can also be set using the python API.  
  * i.e The cuda stream captured from pytorch can be passed into ORT-NV TensorRT RTX EP. Click below to check sample code:

    <Details>

 
    ```python
    import onnxruntime as ort
    import torch
    ...
    sess = ort.InferenceSession('model.onnx')
    if torch.cuda.is_available():
        s = torch.cuda.Stream()
        provider_options = {
          'device_id': 0,
          'user_compute_stream': str(s.cuda_stream)
        }

        sess = ort.InferenceSession(
          model_path,
          providers=[('NvTensorRtRtxExecutionProvider', provider_options)]
        )

        options = sess.get_provider_options()
        assert "NvTensorRtRtxExecutionProvider" in options
        assert options["NvTensorRtRtxExecutionProvider"].get("user_compute_stream", "") == str(s.cuda_stream)
    ...
    ```
    
    </Details>

* To take advantage of user compute stream, it is recommended to use [I/O Binding](https://onnxruntime.ai/docs/performance/device-tensor.html) to bind inputs and outputs to tensors in device.

##### nv_max_workspace_size

* Description: maximum workspace size in bytes for TensorRT RTX engine.

* Default value: 0 (lets TensorRT pick the optimal).

##### nv_dump_subgraphs

* Description: dumps the subgraphs if the ONNX was split across multiple execution providers. 
  * This can help debugging subgraphs, e.g. by using  `trtexec --onnx subgraph_1.onnx` and check the outputs of the parser.

##### nv_detailed_build_log

* Description: enable detailed build step logging on NV TensorRT RTX EP with timing for each engine build.

##### nv_cuda_graph_enable

* Description: this will capture a [CUDA graph](https://developer.nvidia.com/blog/cuda-graphs/) which can drastically help for a network with many small layers as it reduces launch overhead on the CPU.

##### nv_profile_min_shapes

##### nv_profile_max_shapes

##### nv_profile_opt_shapes

* Description: build with explicit dynamic shapes using a profile with the min/max/opt shapes provided.
  * By default TensorRT RTX engines will support dynamic shapes, for perofmance improvements it is possible to specify one or multiple explicit ranges of shapes.
  * The format of the profile shapes is `input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...`
    * These three flags should all be provided in order to enable explicit profile shapes feature.
  * Note that multiple TensorRT RTX profiles can be enabled by passing multiple shapes for the same input tensor.
  * Check TensorRT doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles) for more details.

## NV TensorRT RTX EP Caches
There are two major TRT RTX EP caches:
* Embedded engine model / EPContext model
* Internal TensorRT RTX cache

The internal TensorRT RTX cache is automatically managed by the EP. The user only needs to manage EPContext caching. 
**Caching is important to help reduce session creation time drastically.**

TensorRT RTX separates compilation into an ahead of time (AOT) compiled engine and a just in time (JIT) compilation. The AOT compilation can be stored as EPcontext model, this model will be compatible across multiple GPU generations.
Upon loading such an EPcontext model TensorRT RTX will just in time compile the engine to fit to the used GPU. This JIT process is accelerated by TensorRT RTX's internal cache.
For an example usage see:
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/nv_tensorrt_rtx/nv_basic_test.cc

### More about Embedded engine model / EPContext model
* TODO: decide on a plan for using weight-stripped engines by default. Fix the EP implementation to enable that. Explain the motivation and provide example on how to use the right options in this document.
* EPContext models also **enable packaging an externally compiled engine** using e.g. `trtexec`. A [python script](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py) that is capable of packaging such a precompiled engine into an ONNX file is included in the python tools. (TODO: document how this works with weight-stripped engines).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](./../performance/tune-performance/index.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e nvtensorrttrx`.


### TensorRT RTX Plugins Support
TensorRT RTX doesn't support plugins.
