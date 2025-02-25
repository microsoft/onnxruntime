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

Pre-built binaries of ONNX Runtime with CUDA EP are published for most language bindings. Please
reference [Install ORT](../install).

## Build from source
See [Build instructions](../build/eps.html#cuda).

## Requirements

Please reference table below for official GPU packages dependencies for the ONNX Runtime inferencing package. Note that
ONNX Runtime Training is aligned with PyTorch CUDA versions; refer to the Optimize Training tab on [onnxruntime.ai](https://onnxruntime.ai/getting-started) for supported versions.

Because of [Nvidia CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility), ONNX Runtime built with CUDA 11.8 are compatible with any CUDA 11.x version; ONNX Runtime built with CUDA 12.x are compatible with any CUDA 12.x version.

ONNX Runtime built with cuDNN 8.x is not compatible with cuDNN 9.x, and vice versa. You can choose the package based on CUDA and cuDNN major versions that match your runtime environment (e.g., PyTorch 2.3 uses cuDNN 8.x, while PyTorch 2.4 or later uses cuDNN 9.x).

Note: Starting with version 1.19, **CUDA 12.x** becomes the default version when distributing [ONNX Runtime GPU packages](https://pypi.org/project/onnxruntime-gpu/) in PyPI.

To reduce the need for manual installations of CUDA and cuDNN, and ensure seamless integration between ONNX Runtime and PyTorch, the `onnxruntime-gpu` Python package offers API to load CUDA and cuDNN dynamic link libraries (DLLs) appropriately. For more details, refer to the [Compatibility with PyTorch](#compatibility-with-pytorch) and [Preload DLLs](#preload-dlls) sections.

### CUDA 12.x

| ONNX Runtime  | CUDA   | cuDNN | Notes                                                                |
|---------------|--------|-------|----------------------------------------------------------------------|
| 1.20.x        | 12.x   | 9.x   | Avaiable in PyPI. Compatible with PyTorch >= 2.4.0 for CUDA 12.x.    | 
| 1.19.x        | 12.x   | 9.x   | Avaiable in PyPI. Compatible with PyTorch >= 2.4.0 for CUDA 12.x.    | 
| 1.18.1        | 12.x   | 9.x   | cuDNN 9 is required. No Java package.                                | 
| 1.18.0        | 12.x   | 8.x   | Java package is added.                                               |
| 1.17.x        | 12.x   | 8.x   | Only C++/C# Nuget and Python packages are released. No Java package. |

### CUDA 11.x

| ONNX Runtime         | CUDA   | cuDNN                                   | Notes                                                                                                                                       |
|----------------------|--------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 1.20.x               | 11.8   | 8.x                                     | Not available in PyPI. See [Install ORT](../install) for details. Compatible with PyTorch <= 2.3.1 for CUDA 11.8.                           |
| 1.19.x               | 11.8   | 8.x                                     | Not available in PyPI. See [Install ORT](../install) for details. Compatible with PyTorch <= 2.3.1 for CUDA 11.8.                           |
| 1.18.x               | 11.8   | 8.x                                     | Available in PyPI.                                                                                                                          |
| 1.17<br>1.16<br>1.15 | 11.8   | 8.2.4 (Linux)<br/>8.5.0.96 (Windows)    | Tested with CUDA versions from 11.6 up to 11.8, and cuDNN from 8.2 up to 8.9                                                                |
| 1.14<br>1.13         | 11.6   | 8.2.4 (Linux)<br/>8.5.0.96 (Windows)    | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.5.2<br/>libcublas 11.6.5.2<br/>libcudnn 8.2.4       |
| 1.12<br>1.11         | 11.4   | 8.2.4 (Linux)<br/>8.2.2.26 (Windows)    | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.5.2<br/>libcublas 11.6.5.2<br/>libcudnn 8.2.4       |
| 1.10                 | 11.4   | 8.2.4 (Linux)<br/>8.2.2.26 (Windows)    | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4     |
| 1.9                  | 11.4   | 8.2.4 (Linux)<br/>8.2.2.26 (Windows)    | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4     |
| 1.8                  | 11.0.3 | 8.0.4 (Linux)<br/>8.0.2.39 (Windows)    | libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4  |
| 1.7                  | 11.0.3 | 8.0.4 (Linux)<br/>8.0.2.39 (Windows)    | libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4  |

### CUDA 10.x

| ONNX Runtime | CUDA   | cuDNN | Notes                                                                                                             |
|--------------|--------|-------|-------------------------------------------------------------------------------------------------------------------|
| 1.5-1.6      | 10.2   | 8.0.3 | CUDA 11 can be built from source                                                                                  |
| 1.2-1.4      | 10.1   | 7.6.5 | Requires cublas10-10.2.1.243; cublas 10.1.x will not work                                                         |
| 1.0-1.1      | 10.0   | 7.6.4 | CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017 |

For older versions, please reference the readme and build pages on the release branch.

## Build

For build instructions, please see the [BUILD page](../build/eps.md#cuda).

## Compatibility with PyTorch

The `onnxruntime-gpu` package is designed to work seamlessly with [PyTorch](https://pytorch.org/), provided both are built against the same major version of CUDA and cuDNN. When installing PyTorch with CUDA support, the necessary CUDA and cuDNN DLLs are included, eliminating the need for separate installations of the CUDA toolkit or cuDNN.

To ensure ONNX Runtime utilizes the DLLs installed by PyTorch, you can preload these libraries before creating an inference session. This can be achieved by either importing PyTorch or by using the `onnxruntime.preload_dlls()` function.

**Example 1: Importing PyTorch**

```python
# Import torch will preload necessary DLLs. It need to be done before creating session.
import torch
import onnxruntime

# Create an inference session with CUDA execution provider
session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
```


**Example 2: Using `preload_dlls` Function**

```python
import onnxruntime

# Preload necessary DLLs
onnxruntime.preload_dlls()

# Create an inference session with CUDA execution provider
session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
```

## Preload DLLs

Since version 1.21.0, the `onnxruntime-gpu` package provides the `preload_dlls` function to preload CUDA, cuDNN, and Microsoft Visual C++ (MSVC) runtime DLLs. This function offers flexibility in specifying which libraries to load and from which directories.

**Function Signature:**

```python
onnxruntime.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)
```


**Parameters:**

- `cuda` (bool): Preload CUDA DLLs if set to `True`.
- `cudnn` (bool): Preload cuDNN DLLs if set to `True`.
- `msvc` (bool): Preload MSVC runtime DLLs if set to `True`.
- `directory` (str or None): Directory to load the DLLs from.
  - `None`: Search in default directories.
  - `""` (empty string): Search in NVIDIA site packages.
  - Specific path: Load DLLs from the specified directory.

**Default Search Order:**

When `directory=None`, the function searches for CUDA and cuDNN DLLs in the following order:

1. On Windows, the `lib` directory under the PyTorch installation.
2. Python site-packages directories for NVIDIA CUDA or cuDNN libraries (e.g., `nvidia_cuda_runtime_cu12`, `nvidia_cudnn_cu12`).
3. Fallback to the default DLL loading behavior.

By preloading the necessary DLLs using default search order, you can ensure that ONNX Runtime operates seamlessly with PyTorch.

**Installing CUDA and cuDNN via `onnxruntime-gpu`:**

You can install the necessary CUDA and cuDNN runtime DLLs alongside the `onnxruntime-gpu` package using pip:

```bash
pip install onnxruntime-gpu[cuda,cudnn]
```


**Preloading DLLs from NVIDIA Site Packages:**

To preload CUDA and cuDNN DLLs from NVIDIA site packages and display debug information:

```python
import onnxruntime

# Preload DLLs from NVIDIA site packages
onnxruntime.preload_dlls(directory="")

# Print debug information
onnxruntime.print_debug_info()
```


**Loading DLLs from Specific Directories:**

To load DLLs from a specified location, set the `directory` parameter to an absolute path or a path relative to the ONNX Runtime package root.

**Example: Loading CUDA from System Installation and cuDNN from NVIDIA Site Package**

```python
import os
import onnxruntime

# Load CUDA DLLs from system installation
cuda_path = os.path.join(os.environ["CUDA_PATH"], "bin")
onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=cuda_path)

# Load cuDNN DLLs from NVIDIA site package
onnxruntime.preload_dlls(cuda=False, cudnn=True, directory="..\\nvidia\\cudnn\\bin")

# Print debug information
onnxruntime.print_debug_info()
```

## Configuration Options

The CUDA Execution Provider supports the following configuration options.

### device_id

The device ID.

Default value: 0

### user_compute_stream

Defines the compute stream for the inference to run on.
It implicitly sets the `has_user_compute_stream` option. It cannot be set through `UpdateCUDAProviderOptions`, but
rather `UpdateCUDAProviderOptionsWithValue`.
This cannot be used in combination with an external allocator.

Example python usage:

```python
providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
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

### use_ep_level_unified_stream

Uses the same CUDA stream for all threads of the CUDA EP. This is implicitly enabled
by `has_user_compute_stream`, `enable_cuda_graph` or when using an external allocator.

Default value: false

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

### cudnn_conv_algo_search

The type of search done for cuDNN convolution algorithms.

 Value          | Description                                                                     
----------------|---------------------------------------------------------------------------------
 EXHAUSTIVE (0) | expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx  
 HEURISTIC (1)  | lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7 
 DEFAULT (2)    | default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM        

Default value: EXHAUSTIVE

### cudnn_conv_use_max_workspace

Check [tuning performance for convolution heavy models](#convolution-heavy-models) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API.(sample below)

Default value: 1, for versions 1.14 and later
0, for previous versions

### cudnn_conv1d_pad_to_nc1d

Check [convolution input padding in the CUDA EP](#convolution-input-padding) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API. (sample below)

Default value: 0

### enable_cuda_graph

Check [using CUDA Graphs in the CUDA EP](#using-cuda-graphs-preview) for details on what this flag does.
This flag is only supported from the V2 version of the provider options struct when used using the C API. (sample below)

Default value: 0

### enable_skip_layer_norm_strict_mode

Whether to use strict mode in SkipLayerNormalization cuda implementation. The default and recommended setting is false.
If enabled, accuracy improvement and performance drop can be expected.
This flag is only supported from the V2 version of the provider options struct when used using the C API. (sample below)

Default value: 0

### use_tf32

TF32 is a math mode available on NVIDIA GPUs since Ampere. It allows certain float32 matrix multiplications and convolutions to run much faster on tensor cores with [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) reduced precision: float32 inputs are rounded with 10 bits of mantissa and results are accumulated with float32 precision.

Default value: 1

TensorFloat-32 is enabled by default. Starting from ONNX Runtime 1.18, you can use this flag to disable it for an inference session.

Example python usage:

```python
providers = [("CUDAExecutionProvider", {"use_tf32": 0})]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("my_model.onnx", sess_options=sess_options, providers=providers)
```

This flag is only supported from the V2 version of the provider options struct when used using the C API. (sample below)

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

### prefer_nhwc

This option is available since ONNX Runtime 1.20, where the build has `onnxruntime_USE_CUDA_NHWC_OPS=ON` by default.

If this option is enabled, the execution provider prefers NHWC operators over NCHW. Necessary layout transformations will be applied to the model automatically. Since NVIDIA tensor cores operate more efficiently with NHWC layout, enabling this option can improve performance when the model consists of many supported operators and does not require excessive additional transpose operations. Wider operator support for NHWC is planned for future releases.

This flag is supported only in the V2 version of the provider options struct when using the C API. The V2 provider options struct can be created using [CreateCUDAProviderOptions](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0d29cbf555aa806c050748cf8d2dc172) and updated using [UpdateCUDAProviderOptions](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a4710fc51f75a4b9a75bde20acbfa0783).

Default value: 0

## Performance Tuning

The [I/O Binding feature](../performance/tune-performance/iobinding.md) should be utilized to avoid overhead resulting
from copies on inputs and outputs. Ideally up and downloads for inputs can be hidden behind the inference. This can be
achieved by doing asynchronous copies while running inference. This is demonstrated in
this [PR](https://github.com/microsoft/onnxruntime/pull/14088)

```c++
Ort::RunOptions run_options;
run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
session->Run(run_options, io_binding);
```

By disabling the synchronization on the inference the user has to take care of synchronizing the compute stream after
execution.
This feature should only be used with device local memory or an ORT Value allocated
in [pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/), otherwise the issued
download will be blocking and not behave as desired.

### Convolution-heavy models

ORT leverages CuDNN for convolution operations and the first step in this process is to determine which "optimal"
convolution algorithm to use while performing the convolution operation for the given input configuration (input shape,
filter shape, etc.) in each `Conv` node. This sub-step involves querying CuDNN for a "workspace" memory size and have
this allocated so that CuDNN can use this auxiliary memory while determining the "optimal" convolution algorithm to use.

The default value of `cudnn_conv_use_max_workspace` is 1 for versions 1.14 or later, and 0 for previous versions. When
its value is 0, ORT clamps the workspace size to 32 MB which may lead to a sub-optimal convolution algorithm getting
picked by CuDNN. To allow ORT to allocate the maximum possible workspace as determined by CuDNN, a provider option
named `cudnn_conv_use_max_workspace` needs to get set (as shown below).

Keep in mind that using this flag may increase the peak memory usage by a factor (sometimes a few GBs) but this does
help CuDNN pick the best convolution algorithm for the given input. We have found that this is an important flag to use
while using an fp16 model as this allows CuDNN to pick tensor core algorithms for the convolution operations (if the
hardware supports tensor core operations). This flag may or may not result in performance gains for other data
types (`float` and `double`).

* Python
    ```python
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_conv_heavy_fp16_model.onnx", sess_options=sess_options, providers=providers)
    ```


* C/C++
    ```c++
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    CreateCUDAProviderOptions(&cuda_options);

    std::vector<const char*> keys{"cudnn_conv_use_max_workspace"};
    std::vector<const char*> values{"1"};

    UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

    OrtSessionOptions* session_options = /* ... */;
    SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

    // Finally, don't forget to release the provider options
    ReleaseCUDAProviderOptions(cuda_options);
    ```

* C#
   ```csharp
   var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

   var providerOptionsDict = new Dictionary<string, string>();
   providerOptionsDict["cudnn_conv_use_max_workspace"] = "1";

   cudaProviderOptions.UpdateOptions(providerOptionsDict);

   SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
   ```

### Convolution Input Padding

ORT leverages CuDNN for convolution operations. While CuDNN only takes 4-D or 5-D tensor as input for convolution
operations, dimension padding is needed if the input is 3-D tensor. Given an input tensor of shape [N, C, D], it can be
padded to [N, C, D, 1] or [N, C, 1, D]. While both of these two padding ways produce same output, the performance may be
a lot different because different convolution algorithms are selected, especially on some devices such as A100. By
default the input is padded to [N, C, D, 1]. A provider option named `cudnn_conv1d_pad_to_nc1d` needs to get set (as
shown below) if [N, C, 1, D] is preferred.

* Python
    ```python
    providers = [("CUDAExecutionProvider", {"cudnn_conv1d_pad_to_nc1d": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_conv_model.onnx", sess_options=sess_options, providers=providers)
    ```
* C/C++
    ```c++
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    CreateCUDAProviderOptions(&cuda_options);

    std::vector<const char*> keys{"cudnn_conv1d_pad_to_nc1d"};
    std::vector<const char*> values{"1"};

    UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1);

    OrtSessionOptions* session_options = /* ... */;
    SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);

    // Finally, don't forget to release the provider options
    ReleaseCUDAProviderOptions(cuda_options);
    ```
* C#
    ```csharp
    var cudaProviderOptions = new OrtCUDAProviderOptions(); // Dispose this finally

    var providerOptionsDict = new Dictionary<string, string>();
    providerOptionsDict["cudnn_conv1d_pad_to_nc1d"] = "1";

    cudaProviderOptions.UpdateOptions(providerOptionsDict);

    SessionOptions options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);  // Dispose this finally
    ```

### Using CUDA Graphs (Preview)

While using the CUDA EP, ORT supports the usage
of [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) to remove CPU overhead associated with
launching CUDA kernels sequentially. To enable the usage of CUDA Graphs, use the provider options as shown in the samples
below. ORT supports multi-graph capture capability by passing the user specified gpu_graph_id to the run options. 
gpu_graph_id is optional when the session uses one cuda graph. If not set, the default value is 0. If the gpu_graph_id is 
set to -1, cuda graph capture/replay is disabled in that run.

Currently, there are some constraints with regards to using the CUDA Graphs feature:

* Models with control-flow ops (i.e. `If`, `Loop` and `Scan` ops) are not supported.

* Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the CUDA EP.

* The input/output types of models need to be tensors.

* Shapes and addresses of inputs/outputs cannot change across inference calls for the same graph annotation id. Input
  tensors for replay shall be copied to the address of input tensors used in graph capture.

* In multi-graph capture mode, the captured graphs will remain in the session's lifetime and the captured graph deletion
  feature is not supported at the moment.

* By design, [CUDA Graphs](https://developer.nvidia.com/blog/cuda-10-features-revealed/) is designed to read from/write
  to the same CUDA virtual memory addresses during the graph replaying step as it does during the graph capturing step.
  Due to this requirement, usage of this feature requires using IOBinding so as to bind memory which will be used as
  input(s)/output(s) for the CUDA Graph machinery to read from/write to (please see samples below).

* While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the
  corresponding CUDA memory location(s) of the bound `OrtValue` input(s) (please see samples below to see how this can
  be achieved). This is due to the fact that the "graph replay" will require reading inputs from the same CUDA virtual
  memory addresses.

* Multi-threaded usage is currently not supported, i.e. `Run()` MAY NOT be invoked on the same `InferenceSession` object
  from multiple threads while using CUDA Graphs.

NOTE: The very first `Run()` performs a variety of tasks under the hood like making CUDA memory allocations, capturing
the CUDA graph for the model, and then performing a graph replay to ensure that the graph runs. Due to this, the latency
associated with the first `Run()` is bound to be high. Subsequent `Run()`s only perform graph replays of the graph
captured and cached in the first `Run()`.

* Python

    ```python
    providers = [("CUDAExecutionProvider", {"enable_cuda_graph": '1'})]
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession("my_model.onnx", sess_options=sess_options, providers=providers)

    providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
    y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)

    session = onnxrt.InferenceSession("matmul_2.onnx", providers=providers)
    io_binding = session.io_binding()

    # Pass gpu_graph_id to RunOptions through RunConfigs
    ro = onnxrt.RunOptions()
    # gpu_graph_id is optional if the session uses only one cuda graph
    ro.add_run_config_entry("gpu_graph_id", "1")

    # Bind the input and output
    io_binding.bind_ortvalue_input('X', x_ortvalue)
    io_binding.bind_ortvalue_output('Y', y_ortvalue)

    # One regular run for the necessary memory allocation and cuda graph capturing
    session.run_with_iobinding(io_binding, ro)
    expected_y = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
    np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

    # After capturing, CUDA graph replay happens from this Run onwards
    session.run_with_iobinding(io_binding, ro)
    np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

    # Update input and then replay CUDA graph with the updated input
    x_ortvalue.update_inplace(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32))
    session.run_with_iobinding(io_binding, ro)
    ```
* C/C++
    ```c++
    const auto& api = Ort::GetApi();

    struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(const Ort::Allocator* alloc) {
        alloc_ = alloc;
    }

    void operator()(void* ptr) const {
        alloc_->Free(ptr);
    }

    const Ort::Allocator* alloc_;
    };

    // Enable cuda graph in cuda provider option.
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    api.CreateCUDAProviderOptions(&cuda_options);
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
    std::vector<const char*> keys{"enable_cuda_graph"};
    std::vector<const char*> values{"1"};
    api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);

    Ort::SessionOptions session_options;
    api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), rel_cuda_options.get();

    // Pass gpu_graph_id to RunOptions through RunConfigs
    Ort::RunOptions run_option;
    // gpu_graph_id is optional if the session uses only one cuda graph
    run_option.AddConfigEntry("gpu_graph_id", "1");

    // Create IO bound inputs and outputs.
    Ort::Session session(*ort_env, ORT_TSTR("matmul_2.onnx"), session_options);
    Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Allocator cuda_allocator(session, info_cuda);

    const std::array<int64_t, 2> x_shape = {3, 2};
    std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
                                                            CudaMemoryDeleter(&cuda_allocator));
    cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

    // Create an OrtValue tensor backed by data on CUDA memory
    Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

    const std::array<int64_t, 2> expected_y_shape = {3, 2};
    std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
                                                                CudaMemoryDeleter(&cuda_allocator));

    // Create an OrtValue tensor backed by data on CUDA memory
    Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

    Ort::IoBinding binding(session);
    binding.BindInput("X", bound_x);
    binding.BindOutput("Y", bound_y);

    // One regular run for necessary memory allocation and graph capturing
    session.Run(run_option, binding);

    // After capturing, CUDA graph replay happens from this Run onwards
    session.Run(run_option, binding);

    // Update input and then replay CUDA graph with the updated input
    x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);
    session.Run(run_option, binding);
    ```

* C# (future)

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

cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);
// this implicitly sets "has_user_compute_stream"
UpdateCUDAProviderOptionsWithValue(cuda_options, "user_compute_stream", cuda_stream);
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

Also see the tutorial here on how to [configure CUDA for C# on Windows](../tutorials/csharp/csharp-gpu.md).

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
