---
title: NVIDIA - CUDA
description: Instructions to execute ONNX Runtime applications with CUDA
parent: Execution Providers
nav_order: 1
redirect_from: /docs/reference/execution-providers/CUDA-ExecutionProvider
---

# CUDA Execution Provider
{: .no_toc }

The CUDA Execution Provider enables hardware accelerated computation on Nvidia CUDA-enabled GPUs.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Pre-built binaries of ONNX Runtime with CUDA EP are published for most language bindings. Please reference [Install ORT](../install).

## Requirements

Please reference table below for official GPU packages dependencies for the ONNX Runtime inferencing package. Note that ONNX Runtime Training is aligned with PyTorch CUDA versions; refer to the Training tab on [onnxruntime.ai](https://onnxruntime.ai/) for supported versions. 

Note: Because of CUDA Minor Version Compatibility, Onnx Runtime built with CUDA 11.4 should be compatible with any CUDA 11.x version.
Please reference [Nvidia CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility).

|ONNX Runtime|CUDA|cuDNN|Notes|
|---|---|---|---|
|1.13|11.6|8.2.4 (Linux)<br/>8.5.0.96 (Windows)|libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.5.2<br/>libcublas 11.6.5.2<br/>libcudnn 8.2.4|
|1.12<br/>1.11|11.4|8.2.4 (Linux)<br/>8.2.2.26 (Windows)|libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.5.2<br/>libcublas 11.6.5.2<br/>libcudnn 8.2.4|
|1.10|11.4|8.2.4 (Linux)<br/>8.2.2.26 (Windows)|libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4|
|1.9|11.4|8.2.4 (Linux)<br/>8.2.2.26 (Windows)|libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4|
|1.8|11.0.3|8.0.4 (Linux)<br/>8.0.2.39 (Windows)|libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4|
|1.7|11.0.3|8.0.4 (Linux)<br/>8.0.2.39 (Windows)|libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4|
|1.5-1.6|10.2|8.0.3|CUDA 11 can be built from source|
|1.2-1.4|10.1|7.6.5|Requires cublas10-10.2.1.243; cublas 10.1.x will not work|
|1.0-1.1|10.0|7.6.4|CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017|

For older versions, please reference the readme and build pages on the release branch.

## Build
For build instructions, please see the [BUILD page](../build/eps.md#cuda).

## Configuration Options
The CUDA Execution Provider supports the following configuration options.

### device_id
The device ID.

Default value: 0

### gpu_mem_limit
The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The total device memory usage may be higher.
s: max value of C++ size_t type (effectively unlimited)

### arena_extend_strategy
The strategy for extending the device memory arena.

Value                   | Description
-|-
kNextPowerOfTwo (0)     | subsequent extensions extend by larger amounts (multiplied by powers of two)
kSameAsRequested (1)    | extend by the requested amount

Default value: kNextPowerOfTwo

### cudnn_conv_algo_search
The type of search done for cuDNN convolution algorithms.

Value           | Description
-|-
EXHAUSTIVE (0)  | expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
HEURISTIC (1)   | lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
DEFAULT (2)     | default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

Default value: EXHAUSTIVE

### do_copy_in_default_stream
Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false, there are race conditions and possibly better performance.

Default value: true

### cudnn_conv_use_max_workspace
Check [tuning performance for convolution heavy models](../performance/tune-performance.md#convolution-heavy-models-and-the-cuda-ep) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API. The V2 provider options struct can be created using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0d29cbf555aa806c050748cf8d2dc172) and updated using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a4710fc51f75a4b9a75bde20acbfa0783). Please take a look at the sample below for an example.

Default value: 0

### cudnn_conv1d_pad_to_nc1d
Check [convolution input padding in the CUDA EP](../performance/tune-performance.md#convolution-input-padding-in-the-cuda-ep) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API. The V2 provider options struct can be created using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0d29cbf555aa806c050748cf8d2dc172) and updated using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a4710fc51f75a4b9a75bde20acbfa0783). Please take a look at the sample below for an example.

Default value: 0

### enable_cuda_graph
Check [using CUDA Graphs in the CUDA EP](../performance/tune-performance.md#using-cuda-graphs-in-the-cuda-ep) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API. The V2 provider options struct can be created using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0d29cbf555aa806c050748cf8d2dc172) and updated using [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a4710fc51f75a4b9a75bde20acbfa0783).

Default value: 0

## Samples

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```

### C/C++

#### Using legacy provider options struct

```c++
OrtSessionOptions* session_options = /* ... */;

OrtCUDAProviderOptions options;
options.device_id = 0;
options.arena_extend_strategy = 0;
options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
options.do_copy_in_default_stream = 1;

SessionOptionsAppendExecutionProvider_CUDA(session_options, &options);
```

#### Using V2 provider options struct

```c++
OrtCUDAProviderOptionsV2* cuda_options = nullptr;
CreateCUDAProviderOptions(&cuda_options);

std::vector<const char*> keys{"device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d"};
std::vector<const char*> values{"0", "2147483648", "kSameAsRequested", "DEFAULT", "1", "1", "1"};

UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

OrtSessionOptions* session_options = /* ... */;
SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

// Finally, don't forget to release the provider options
ReleaseCUDAProviderOptions(cuda_options);
```

### C#

```c#
var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

var providerOptionsDict = new Dictionary<string, string>();
providerOptionsDict["device_id"] = "0";
providerOptionsDict["gpu_mem_limit"] = "2147483648";
providerOptionsDict["arena_extend_strategy"] = "kSameAsRequested";
providerOptionsDict["cudnn_conv_algo_search"] = "DEFAULT";
providerOptionsDict["do_copy_in_default_stream"] = "1";
providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";
providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";

cudaProviderOptions.UpdateOptions(providerOptionsDict);

SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
```

### Java

```java
OrtCUDAProviderOptions cudaProviderOptions = new OrtCUDAProviderOptions(/*device id*/0); // Must be closed after the session closes

cudaProviderOptions.add("gpu_mem_limit","2147483648");
cudaProviderOptions.add("arena_extend_strategy","kSameAsRequested");
cudaProviderOptions.add("cudnn_conv_algo_search","DEFAULT");
cudaProviderOptions.add("do_copy_in_default_stream","1");
cudaProviderOptions.add("cudnn_conv_use_max_workspace","1");
cudaProviderOptions.add("cudnn_conv1d_pad_to_nc1d","1");

OrtSession.SessionOptions options = new OrtSession.SessionOptions(); // Must be closed after the session closes
options.addCUDA(cudaProviderOptions);
```

