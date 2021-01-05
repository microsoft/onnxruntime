---
title: CUDA
parent: Execution Providers
grand_parent: Reference
nav_order: 1
---

# CUDA Execution Provider

The CUDA Execution Provider enables hardware accelerated computation on Nvidia CUDA-enabled GPUs.

## Build
For build instructions, please see the [BUILD page](../../how-to/build.md#CUDA).

## Configuration Options
The CUDA Execution Provider supports the following configuration options.

### device_id
The device ID.

### cuda_mem_limit
The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The total device memory usage may be higher.

### arena_extend_strategy
The strategy for extending the device memory arena.

Value                   | Description
-|-
kNextPowerOfTwo (0)     | subsequent extensions extend by larger amounts (multiplied by powers of two)
kSameAsRequested (1)    | extend by the requested amount

### cudnn_conv_algo_search
The type of search done for cuDNN convolution algorithms.

Value           | Description
-|-
EXHAUSTIVE (0)  | expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
HEURISTIC (1)   | lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
DEFAULT (2)     | default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

### do_copy_in_default_stream
Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false, there are race conditions and possibly better performance.

## Example Usage

### Python

```python
import onnxruntime as ort

model_path = 'model.onnx'

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cuda_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```

### C/C++

```c++
OrtSessionOptions* session_options = /* ... */;

OrtCUDAProviderOptions options;
options.device_id = 0;
options.arena_extend_strategy = 0;
options.cuda_mem_limit = 2 * 1024 * 1024 * 1024;
options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
options.do_copy_in_default_stream = 1;

SessionOptionsAppendExecutionProvider_CUDA(session_options, &options);
```
