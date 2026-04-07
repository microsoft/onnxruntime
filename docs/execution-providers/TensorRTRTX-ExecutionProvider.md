---
title: NVIDIA - TensorRT RTX
description: Instructions to execute ONNX Runtime on NVIDIA RTX GPUs with the NVIDIA TensorRT RTX execution provider
parent: Execution Providers
nav_order: 2
redirect_from: /docs/reference/execution-providers/TensorRTRTX-ExecutionProvider
---

# NVIDIA TensorRT RTX Execution Provider
{: .no_toc }

> **⚠️ Deprecation Notice:** The built-in TensorRT RTX Execution Provider in the ONNX Runtime repository is deprecated. We strongly recommend using the standalone EP ABI plugin instead. The open-source TensorRT RTX EP ABI is available at [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI). Instructions for using the standalone EP ABI are included in the [Usage](#usage) section below. All execution provider options supported by the built-in EP are fully supported by the EP ABI plugin.

The NVIDIA TensorRT-RTX Execution Provider (EP) is an inference deployment solution designed specifically for NVIDIA RTX GPUs. It is optimized for client-centric use cases.. 

TensorRT RTX EP provides the following benefits:

* **Small package footprint:** Optimized resource usage on end-user systems at just under 200 MB.  
* **Faster model compile and load times:** Leverages just-in-time compilation techniques, to build RTX hardware-optimized engines on end-user devices in seconds.  
* **Portability:** Seamlessly use cached models across multiple RTX GPUs.

The TensorRT RTX EP leverages NVIDIA’s new deep learning inference engine, [TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx), to accelerate ONNX models on RTX GPUs. Microsoft and NVIDIA collaborated closely to integrate the TensorRT RTX EP with ONNX Runtime.

TensorRT RTX EP supports RTX GPUs based on Ampere and later architectures - NVIDIA GeForce RTX 30xx and above.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Currently, TensorRT RTX EP can be built from the source code. Support for installation from package managers, such as PyPi and NuGet, is coming soon. See the [WinML install section](../install/#cccwinml-installs) for WinML-related installation instructions.

## Build from source

**Standalone EP ABI Plugin (Recommended)**

To build the standalone TensorRT RTX EP ABI plugin, please refer to the build instructions in the [TensorRT RTX EP ABI repository](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI#build-from-source).

**Built-in EP (Deprecated)**

Information on minimum requirements and how to build the built-in EP from source can be found [here](../build/eps.md#nvidia-tensorrt-rtx).

## Usage

### C/C++
```c++
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SampleApp");
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {});
Ort::Session session(env, model_path, session_options);
```

**Using the EP ABI (Standalone Plugin)**

If you are using the TensorRT RTX EP as a standalone plugin via the EP ABI (introduced in ORT 1.23.0), use the V2 device-based EP API:

```cpp
#include <onnxruntime_cxx_api.h>

OrtApi const& ortApi = Ort::GetApi();
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");
Ort::SessionOptions session_options;

// 1. Register the EP plugin library
ortApi.RegisterExecutionProviderLibrary(
    env, "NvTensorRTRTXExecutionProvider",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

// 2. Enumerate available EP devices and find TensorRT RTX
const OrtEpDevice* const* ep_devices = nullptr;
size_t num_ep_devices;
ortApi.GetEpDevices(env, &ep_devices, &num_ep_devices);

const OrtEpDevice* trt_device = nullptr;
for (size_t i = 0; i < num_ep_devices; i++) {
    if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]),
              "NvTensorRTRTXExecutionProvider") == 0) {
        trt_device = ep_devices[i];
        break;
    }
}

// 3. Append the EP with provider options
const char* keys[]   = {"enable_cuda_graph"};
const char* values[] = {"1"};
ortApi.SessionOptionsAppendExecutionProvider_V2(
    session_options, env, &trt_device, 1,
    keys, values, 1);

// 4. Create session
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

### Python

Register the TensorRT RTX  EP by specifying it in the providers argument when creating an InferenceSession.

```python
import onnxruntime as ort
session = ort.InferenceSession(model_path, providers=['NvTensorRtRtxExecutionProvider'])
```

## Features

### CUDA Graph

CUDA Graph is a representation of a sequence of GPU operations, such as kernel launches and memory copies, captured from a CUDA stream. Instead of the CPU launching each operation individually, the entire sequence is captured at once and then replayed on the GPU. This process significantly reduces CPU overhead and improves GPU utilization. Find out more details about CUDA Graphs from [this blog](https://developer.nvidia.com/blog/cuda-graphs/).

**Usage**

CUDA Graph is enabled by default. To completely disable CUDA Graph, you can set the `enable_cuda_graph` provider option to `False` (or `0`).

**Python (Disabling CUDA Graph)**

```python
trt_rtx_provider_options = {'enable_cuda_graph': False}
providers = [('NvTensorRTRTXExecutionProvider', trt_rtx_provider_options)]
session = ort.InferenceSession("model.onnx", providers=providers)
```

**C/C++ (Disabling CUDA Graph)**
```cpp
const auto& api = Ort::GetApi();
Ort::SessionOptions session_options;
const char* keys[]   = {onnxruntime::nv::provider_option_names::kCudaGraphEnable};
const char* values[] = {"0"};
OrtStatus* status = api.SessionOptionsAppendExecutionProvider(session_options, onnxruntime::kNvTensorRTRTXExecutionProvider, keys, values, 1);
Ort::Session session(env, model_path, session_options);
```

**ONNXRuntime Perf Test (Disabling CUDA Graph)**

*Using the EP ABI (Standalone Plugin):*
```sh
onnxruntime_perf_test.exe --plugin_eps nvtensorrtrtx --plugin_ep_libs "nvtensorrtrtx|/path/to/onnxruntime_providers_nv_tensorrt_rtx.dll" -I -t 5 -i "enable_cuda_graph|0" "model.onnx"
```

*Using the Built-in EP (Deprecated):*
```sh
onnxruntime_perf_test.exe -I -t 5 -e nvtensorrtrtx -i "enable_cuda_graph|0" "model.onnx"
```

**Effectively Using CUDA Graphs**

CUDA Graph can be beneficial when execution patterns are static and involve many small GPU kernels. This feature helps reduce CPU overhead and improve GPU utilization, particularly for static execution plans run more than twice.

Avoid enabling CUDA Graph or proceed with caution if:

* Input shapes or device bindings frequently change.  
* The control flow is conditional and data-dependent.


### EP context model

EP context nodes are precompiled optimized formats that are execution provider specific. They enable to compile a standard ONNX model once and make any subsequent load of the same unchanged model as fast as possible.

TensorRT RTX handle compilation into two distinct phases:

* **Ahead-of-Time (AOT)**: The ONNX model is compiled into an optimized binary blob, and stored as an EP context model.  
* **Just-in-Time (JIT)**: At inference time, the EP context model is loaded and TensorRT RTX dynamically compiles the binary blob (engine) to optimize it for the exact GPU hardware being used.

**Generating EP Context Models**

ONNX Runtime 1.22 introduced dedicated [Compile APIs](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/compile_api.h) to simplify the generation of EP context models:

```cpp
// AOT phase
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetInputModelPath(input_model_path);
compile_options.SetOutputModelPath(compile_model_path);

Ort::Status status = Ort::CompileModel(env, compile_options);
```

After successful generation, the EP context model can be directly loaded for inference:

```cpp
// JIT phase
Ort::Session session(env, compile_model_path, session_options);
```

This leads to a considerable reduction in session creation time, improving the overall user experience.

The JIT time can be further improved using runtime cache. A runtime cache directory  with a per model cache is created. This cache stores the compiled CUDA kernels and reduces session load time. Learn more  about the process [here](#runtime-cache).

For a practical example of usage for EP context, please refer to:

* EP context samples  
* EP context [unit tests](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/nv_tensorrt_rtx/nv_ep_context_test.cc)


ONNXRuntime Perf Test can also be used to quick generate an EP context model:

*Using the EP ABI (Standalone Plugin):*
```sh
onnxruntime_perf_test.exe --plugin_eps nvtensorrtrtx --plugin_ep_libs "nvtensorrtrtx|/path/to/onnxruntime_providers_nv_tensorrt_rtx.dll" -I -r 1 --compile_ep_context --compile_model_path "/path/to/model_ctx.onnx" "/path/to/model.onnx"
```

*Using the Built-in EP (Deprecated):*
```sh
onnxruntime_perf_test.exe -e nvtensorrtrtx -I -r 1 --compile_ep_context --compile_model_path "/path/to/model_ctx.onnx" "/path/to/model.onnx"
```

**Python**

```py
import onnxruntime as ort

input_path = "/path/to/model.onnx"
output_path = "/path/to/model_ctx.onnx"

session_options = ort.SessionOptions()
session_options.add_provider("NvTensorRTRTXExecutionProvider", {})
model_compiler = ort.ModelCompiler(session_options, input_path)
model_compiler.compile_to_file(output_path)
```

**NVIDIA recommended settings**

* For models > 2GB, set embed_mode = 0 in model compilation options. If binary blob is embedded within the EP context, it fails for > 2GB models due to protobuf limitations
```cpp
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetEpContextEmbedMode(0);
```


### Runtime cache

Runtime caches help reduce JIT compilation time. When a user compiles an EP context and loads the resulting model for the first time, the system generates specialized CUDA kernels for the GPU. By setting the provider option `"nv_runtime_cache_path"` to a directory, a cache is created for each TensorRT RTX engine in an EP context node. On subsequent loads, this cache allows the system to quickly deserialize precompiled kernels instead of compiling them again. This is especially helpful for large models with many different operators, such as SD 1.5, which includes a mix of Conv and MatMul operations. The cache only contains compiled kernels. No information about the model’s graph structure or weights is stored.


## Execution Provider Options
TensorRT RTX EP provides the following user configurable options with the [Execution Provider Options](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/nv_tensorrt_rtx/nv_provider_options.h)

> **Note:** All the options listed below apply to both the built-in TensorRT RTX EP and the standalone EP ABI plugin.


| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| device_id | `int` | GPU device identifier | 0 |
| user_compute_stream | `str` | Specify compute stream to run GPU workload | "" |
| nv_max_workspace_size | `int` | Maximum TensorRT engine workspace (bytes) | 0 (auto) |
| nv_max_shared_mem_size | `int` | Maximum TensorRT engine workspace (bytes) | 0 (auto) |
| nv_dump_subgraphs | `bool` | Enable subgraph dumping for debugging | false |
| nv_detailed_build_log | `bool` | Enable detailed build logging | false |
| enable_cuda_graph | `bool` | Enable [CUDA graph](https://developer.nvidia.com/blog/cuda-graphs/) to reduce inference overhead. Helpful for smaller models | true |
| nv_profile_min_shapes | `str` | Comma-separated list of input tensor shapes for the minimum optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| nv_profile_max_shapes | `str` | Comma-separated list of input tensor shapes for the maximum optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| nv_profile_opt_shapes | `str` | Comma-separated list of input tensor shapes for the optimal optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| nv_multi_profile_enable | `bool` | Enable support for multiple optimization profiles in TensorRT engine. Allows dynamic input shapes for different inference requests | false |
| nv_use_external_data_initializer | `bool` | Use external data initializer for model weights. Useful for EP context large models with external data files | false |
| nv_runtime_cache_path | `str` | Path to store runtime cache. Setting this enables faster model loading by caching JIT compiled kernels for each TensorRT RTX engine. | "" (disabled) |



Click below for Python API example:


<details>

```python
import onnxruntime as ort

model_path = '/path/to/model'

# note: for bool type options in python API, set them as False/True
provider_options = {
  'device_id': 0,
  'nv_dump_subgraphs': False,
  'nv_detailed_build_log': True,
  'user_compute_stream': stream_handle
}

sesion_options = ort.SessionOptions()
session = ort.InferenceSession(model_path, sess_options=sesion_options, providers=[('NvTensorRTRTXExecutionProvider', provider_options)])
```
</details>


Click below for C++ API example:


<details>

```c++
Ort::SessionOptions session_options;

// define a cuda stream
cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);

char stream_handle[32];
sprintf_s(stream_handle, "%lld", (uint64_t)cuda_stream);

std::unordered_map<std::string, std::string> provider_options;
provider_options[onnxruntime::nv::provider_option_names::kDeviceId] = "1";
provider_options[onnxruntime::nv::provider_option_names::kUserComputeStream] = stream_handle;

session_options.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, provider_options);
```

</details>



> NOTE: For bool type options, assign them with **True**/**False** in python, or **1**/**0** in C++.


#### Profile shape options

* Description: build with explicit dynamic shapes using a profile with the min/max/opt shapes provided.  
  * By default TensorRT RTX engines support dynamic shapes. For additional performance improvements, you can specify one or multiple explicit ranges of shapes.  
  * The format of the profile shapes is `input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...`  
    * These three flags  must be provided in order to enable explicit profile shapes.  
  * Note that multiple TensorRT RTX profiles can be enabled by passing multiple shapes for the same input tensor.  
  * Check TensorRT for RTX doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-dynamic-shapes.html) for more details.

## Performance test

**Using the EP ABI (Standalone Plugin)**

When using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest#onnxruntime-performance-test) with the standalone EP ABI plugin, use the `--plugin_eps` and `--plugin_ep_libs` flags:

```sh
onnxruntime_perf_test.exe --plugin_eps nvtensorrtrtx --plugin_ep_libs "nvtensorrtrtx|/path/to/onnxruntime_providers_nv_tensorrt_rtx.dll" -I -g -t 5 "/path/to/model.onnx" -
```

**Using the Built-in EP (Deprecated)**

When using the built-in EP, use the flag `-e nvtensorrtrtx`:

```sh
onnxruntime_perf_test.exe -e nvtensorrtrtx -I -g -t 5 "/path/to/model.onnx" -
```


