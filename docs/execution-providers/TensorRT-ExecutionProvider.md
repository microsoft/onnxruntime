---
title: NVIDIA - TensorRT
description: Instructions to execute ONNX Runtime on NVIDIA GPUs with the TensorRT execution provider
parent: Execution Providers
nav_order: 2
redirect_from: /docs/reference/execution-providers/TensorRT-ExecutionProvider
---

# TensorRT Execution Provider
{: .no_toc }

With the TensorRT execution provider, the ONNX Runtime delivers better inferencing performance on the same hardware compared to generic GPU acceleration.

The TensorRT execution provider in the ONNX Runtime makes use of NVIDIA's [TensorRT](https://developer.nvidia.com/tensorrt) Deep Learning inferencing engine to accelerate ONNX model in their family of GPUs. Microsoft and NVIDIA worked closely to integrate the TensorRT execution provider with ONNX Runtime.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install
Please select the GPU (CUDA/TensorRT) version of OnnxRuntime: https://onnxruntime.ai/docs/install. Pre-built packages and Docker images are available for Jetpack in the [Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

## Build from source
See [Build instructions](../build/eps.md#tensorrt).

## Requirements

| ONNX Runtime | TensorRT | CUDA   |
|:-------------|:---------|:-------|
| 1.16-main    | 8.6      | 11.8   |
| 1.15         | 8.6      | 11.8   |
| 1.14         | 8.5      | 11.6   |
| 1.12-1.13    | 8.4      | 11.4   |
| 1.11         | 8.2      | 11.4   |
| 1.10         | 8.0      | 11.4   |
| 1.9          | 8.0      | 11.4   |
| 1.7-1.8      | 7.2      | 11.0.3 |
| 1.5-1.6      | 7.1      | 10.2   |
| 1.2-1.4      | 7.0      | 10.1   |
| 1.0-1.1      | 6.0      | 10.0   |

For more details on CUDA/cuDNN versions, please see [CUDA EP requirements](./CUDA-ExecutionProvider.md#requirements).

## Usage
### C/C++
```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
Ort::Session session(env, model_path, sf);
```

The C API details are [here](../get-started/with-c.md).

### Python
To use TensorRT execution provider, you must explicitly register TensorRT execution provider when instantiating the `InferenceSession`.
Note that it is recommended you also register `CUDAExecutionProvider` to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.

```python
import onnxruntime as ort
# set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
sess = ort.InferenceSession('model.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
```

## Configurations
There are two ways to configure TensorRT settings, either by **TensorRT Execution Provider Session Options** or **Environment Variables(deprecated)** shown as below:


| TensorRT EP Session Options           | Environment Variables(deprecated)              | Type   |
|:--------------------------------------|:-----------------------------------------------|:-------|
| trt_max_workspace_size                | ORT_TENSORRT_MAX_WORKSPACE_SIZE                | int    |
| trt_max_partition_iterations          | ORT_TENSORRT_MAX_PARTITION_ITERATIONS          | int    |
| trt_min_subgraph_size                 | ORT_TENSORRT_MIN_SUBGRAPH_SIZE                 | int    |
| trt_fp16_enable                       | ORT_TENSORRT_FP16_ENABLE                       | bool   |
| trt_int8_enable                       | ORT_TENSORRT_INT8_ENABLE                       | bool   |
| trt_int8_calibration_table_name       | ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME       | string |
| trt_int8_use_native_calibration_table | ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE | bool   |
| trt_dla_enable                        | ORT_TENSORRT_DLA_ENABLE                        | bool   |
| trt_dla_core                          | ORT_TENSORRT_DLA_CORE                          | int    |
| trt_engine_cache_enable               | ORT_TENSORRT_ENGINE_CACHE_ENABLE               | bool   |
| trt_engine_cache_path                 | ORT_TENSORRT_CACHE_PATH                        | string |
| trt_dump_subgraphs                    | ORT_TENSORRT_DUMP_SUBGRAPHS                    | bool   |
| trt_force_sequential_engine_build     | ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD     | bool   |
| trt_context_memory_sharing_enable     | ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE     | bool   |
| trt_layer_norm_fp32_fallback          | ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK          | bool   |
| trt_timing_cache_enable               | ORT_TENSORRT_TIMING_CACHE_ENABLE               | bool   |
| trt_force_timing_cache                | ORT_TENSORRT_FORCE_TIMING_CACHE_ENABLE         | bool   |
| trt_detailed_build_log                | ORT_TENSORRT_DETAILED_BUILD_LOG_ENABLE         | bool   |
| trt_build_heuristics_enable           | ORT_TENSORRT_BUILD_HEURISTICS_ENABLE           | bool   |
| trt_sparsity_enable                   | ORT_TENSORRT_SPARSITY_ENABLE                   | bool   |
| trt_builder_optimization_level        | ORT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL        | int    |
| trt_auxiliary_streams                 | ORT_TENSORRT_AUXILIARY_STREAMS                 | int    |
| trt_tactic_sources                    | ORT_TENSORRT_TACTIC_SOURCES                    | string |
| trt_extra_plugin_lib_paths            | ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS            | string |
| trt_profile_min_shapes                | ORT_TENSORRT_PROFILE_MIN_SHAPES                | string |
| trt_profile_max_shapes                | ORT_TENSORRT_PROFILE_MAX_SHAPES                | string |
| trt_profile_opt_shapes                | ORT_TENSORRT_PROFILE_OPT_SHAPES                | string |

> Note: for bool type options, assign them with **True**/**False** in python, or **1**/**0** in C++.

### Execution Provider Options

TensorRT configurations can be set by execution provider options. It's useful when each model and inference session have their own configurations. In this case, execution provider option settings will override any environment variable settings. All configurations should be set explicitly, otherwise default value will be taken.

* `device_id`: The device ID.
    * Default value: 0

* `user_compute_stream`: Defines the compute stream for the inference to run on. It implicitly sets the `has_user_compute_stream` option. It cannot be set through `UpdateTensorRTProviderOptions`, but rather `UpdateTensorRTProviderOptionsWithValue`.
    * This cannot be used in combination with an external allocator.
    * This can not be set using the python API.

* `trt_max_workspace_size`: maximum workspace size for TensorRT engine.
    * Default value: 1073741824 (1GB).

* `trt_max_partition_iterations`: maximum number of iterations allowed in model partitioning for TensorRT.
    * If target model can't be successfully partitioned when the maximum number of iterations is reached, the whole model will fall back to other execution providers such as CUDA or CPU.
    * Default value: 1000.

* `trt_min_subgraph_size`: minimum node size in a subgraph after partitioning.
  * Subgraphs with smaller size will fall back to other execution providers.
  * Default value: 1.

* `trt_fp16_enable`: Enable FP16 mode in TensorRT.
    > Note: not all Nvidia GPUs support FP16 precision.

* `trt_int8_enable`: Enable INT8 mode in TensorRT.
    > Note:  not all Nvidia GPUs support INT8 precision.

* `trt_int8_calibration_table_name`: Specify INT8 calibration table file for non-QDQ models in INT8 mode.
    > Note: calibration table should not be provided for QDQ model because TensorRT doesn't allow calibration table to be loded if there is any Q/DQ node in the model. By default the name is empty.

* `trt_int8_use_native_calibration_table`: Select what calibration table is used for non-QDQ models in INT8 mode.
  * If `True`, native TensorRT generated calibration table is used;
  * If `False`, ONNXRUNTIME tool generated calibration table is used.
  > Note: Please copy up-to-date calibration table file to `trt_engine_cache_path` before inference. Calibration table is specific to models and calibration data sets. Whenever new calibration table is generated, old file in the path should be cleaned up or be replaced.

* `trt_dla_enable`: Enable DLA (Deep Learning Accelerator).
    > Note: Not all Nvidia GPUs support DLA.

* `trt_dla_core`: Specify DLA core to execute on. Default value: 0.


* `trt_engine_cache_enable`: Enable TensorRT engine caching.

    * The purpose of using engine caching is to save engine build time in the case that TensorRT may take long time to optimize and build engine.

    * Engine will be cached when it's built for the first time so next time when new inference session is created the engine can be loaded directly from cache. In order to validate that the loaded engine is usable for current inference, engine profile is also cached and loaded along with engine. If current input shapes are in the range of the engine profile, the loaded engine can be safely used. Otherwise if input shapes are out of range, profile cache will be updated to cover the new shape and engine will be recreated based on the new profile (and also refreshed in the engine cache).

        * Note each engine is created for specific settings such as model path/name, precision (FP32/FP16/INT8 etc), workspace, profiles etc, and specific GPUs and it's not portable, so it's essential to make sure those settings are not changing, otherwise the engine needs to be rebuilt and cached again.

  > **Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:**
  >
  >    * Model changes (if there are any changes to the model topology, opset version, operators etc.)
  >    * ORT version changes (i.e. moving from ORT version 1.8 to 1.9)
  >    * TensorRT version changes (i.e. moving from TensorRT 7.0 to 8.0)

* `trt_engine_cache_path`: Specify path for TensorRT engine and profile files if `trt_engine_cache_enable` is `True`, or path for INT8 calibration table file if `trt_int8_enable` is `True`.

* `trt_dump_subgraphs`: Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem.
  * This can help debugging subgraphs, e.g. by using  `trtexec --onnx my_model.onnx` and check the outputs of the parser.

* `trt_force_sequential_engine_build`: Sequentially build TensorRT engines across provider instances in multi-GPU environment.

* `trt_context_memory_sharing_enable`: Share execution context memory between TensorRT subgraphs.

* `trt_layer_norm_fp32_fallback`: Force Pow + Reduce ops in layer norm to FP32.

* `trt_timing_cache_enable`: Enable TensorRT timing cache.
  * Check [Timing cache](#timing-cache) for details.

* `trt_force_timing_cache`: Force the TensorRT timing cache to be used even if device profile does not match.
    * A perfect match is only the exact same GPU model as the on that produced the timing cache.

* `trt_detailed_build_log`: Enable detailed build step logging on TensorRT EP with timing for each engine build.

* `trt_build_heuristics_enable`: Build engine using heuristics to reduce build time.

* `trt_sparsity_enable`: Control if sparsity can be used by TRT.
  * Check `--sparsity` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `trt_builder_optimization_level`: Set the builder optimization level.
  > WARNING: levels below 3 do not guarantee good engine performance, but greatly improve build time.  Default 3, valid range [0-5]. Check `--builderOptimizationLevel` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `trt_auxiliary_streams`: Set maximum number of auxiliary streams per inference stream.
  * Setting this value to 0 will lead to optimal memory usage.
  * Default -1 = heuristics.
  * Check `--maxAuxStreams` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `trt_tactic_sources`: Specify the tactics to be used by adding (+) or removing (-) tactics from the default tactic sources (default = all available tactics)
  * e.g. "-CUDNN,+CUBLAS" available keys: "CUBLAS", "CUBLAS_LT", "CUDNN" or "EDGE_MASK_CONVOLUTIONS".

* `trt_extra_plugin_lib_paths`: Specify extra TensorRT plugin library paths.
  * ORT TRT by default supports any TRT plugins registered in TRT registry in TRT plugin library (i.e., `libnvinfer_plugin.so`).
  * Moreover, if users want to use other TRT plugins that are not in TRT plugin library,
    * for example, FasterTransformer has many TRT plugin implementations for different models, user can specify like this `ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS=libvit_plugin.so;libvit_int8_plugin.so`.

* `trt_profile_min_shapes`, `trt_profile_max_shapes` and `trt_profile_opt_shapes` : Build with dynamic shapes using a profile with the min/max/opt shapes provided.
  * The format of the profile shapes is `input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...`
    * These three flags should all be provided in order to enable explicit profile shapes feature.
  * Check [Explicit shape range for dynamic shape input](#explicit-shape-range-for-dynamic-shape-input) and TRT doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles) for more details.


Besides, `device_id` can also be set by execution provider option.

#### Click below for C++ API example:

<details>

```c++
Ort::SessionOptions session_options;

const auto& api = Ort::GetApi();
OrtTensorRTProviderOptionsV2* tensorrt_options;
Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

std::vector<const char*> option_keys = {
    "device_id",
    "trt_max_workspace_size",
    "trt_max_partition_iterations",
    "trt_min_subgraph_size",
    "trt_fp16_enable",
    "trt_int8_enable",
    "trt_int8_use_native_calibration_table",
    "trt_dump_subgraphs",
    // below options are strongly recommended !
    "trt_engine_cache_enable",
    "trt_engine_cache_path",
    "trt_timing_cache_enable",
    "trt_timing_cache_path",
};
std::vector<const char*> option_values = {
    "1",
    "2147483648",
    "10",
    "5",
    "1",
    "1",
    "1",
    "1",
    "1",
    "1",
    "/path/to/cache",
    "1",
    "/path/to/cache", // can be same as the engine cache folder
};

Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options,
                                                    option_keys.data(), option_values.data(), option_keys.size()));


cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);
// this implicitly sets "has_user_compute_stream"
Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(cuda_options, "user_compute_stream", cuda_stream))

session_options.AppendExecutionProvider_TensorRT_V2(tensorrt_options);
/// below code can be used to print all options
OrtAllocator* allocator;
char* options;
Ort::ThrowOnError(api.GetAllocatorWithDefaultOptions(&allocator));
Ort::ThrowOnError(api.GetTensorRTProviderOptionsAsString(tensorrt_options,          allocator, &options));

```

</details>

#### Click below for Python API example:

<details>

```python
import onnxruntime as ort

model_path = '<path to model>'

# note: for bool type options in python API, set them as False/True
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
]

sess_opt = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=providers)
```

</details>

### Environment Variables(deprecated)

Following environment variables can be set for TensorRT execution provider. Click below for more details.

<details>

* `ORT_TENSORRT_MAX_WORKSPACE_SIZE`: maximum workspace size for TensorRT engine. Default value: 1073741824 (1GB).

* `ORT_TENSORRT_MAX_PARTITION_ITERATIONS`: maximum number of iterations allowed in model partitioning for TensorRT. If target model can't be successfully partitioned when the maximum number of iterations is reached, the whole model will fall back to other execution providers such as CUDA or CPU. Default value: 1000.

* `ORT_TENSORRT_MIN_SUBGRAPH_SIZE`: minimum node size in a subgraph after partitioning. Subgraphs with smaller size will fall back to other execution providers. Default value: 1.

* `ORT_TENSORRT_FP16_ENABLE`: Enable FP16 mode in TensorRT. 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support FP16 precision.

* `ORT_TENSORRT_INT8_ENABLE`: Enable INT8 mode in TensorRT. 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support INT8 precision.

* `ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME`: Specify INT8 calibration table file for non-QDQ models in INT8 mode. Note calibration table should not be provided for QDQ model because TensorRT doesn't allow calibration table to be loded if there is any Q/DQ node in the model. By default the name is empty.

* `ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE`: Select what calibration table is used for non-QDQ models in INT8 mode. If 1, native TensorRT generated calibration table is used; if 0, ONNXRUNTIME tool generated calibration table is used. Default value: 0.
    * **Note: Please copy up-to-date calibration table file to `ORT_TENSORRT_CACHE_PATH` before inference. Calibration table is specific to models and calibration data sets. Whenever new calibration table is generated, old file in the path should be cleaned up or be replaced.**

* `ORT_TENSORRT_DLA_ENABLE`: Enable DLA (Deep Learning Accelerator). 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support DLA.

* `ORT_TENSORRT_DLA_CORE`: Specify DLA core to execute on. Default value: 0.

* `ORT_TENSORRT_ENGINE_CACHE_ENABLE`: Enable TensorRT engine caching. The purpose of using engine caching is to save engine build time in the case that TensorRT may take long time to optimize and build engine. Engine will be cached when it's built for the first time so next time when new inference session is created the engine can be loaded directly from cache. In order to validate that the loaded engine is usable for current inference, engine profile is also cached and loaded along with engine. If current input shapes are in the range of the engine profile, the loaded engine can be safely used. Otherwise if input shapes are out of range, profile cache will be updated to cover the new shape and engine will be recreated based on the new profile (and also refreshed in the engine cache). Note each engine is created for specific settings such as model path/name, precision (FP32/FP16/INT8 etc), workspace, profiles etc, and specific GPUs and it's not portable, so it's essential to make sure those settings are not changing, otherwise the engine needs to be rebuilt and cached again. 1: enabled, 0: disabled. Default value: 0.
    * **Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:**
        * Model changes (if there are any changes to the model topology, opset version, operators etc.)
        * ORT version changes (i.e. moving from ORT version 1.8 to 1.9)
        * TensorRT version changes (i.e. moving from TensorRT 7.0 to 8.0)
        * Hardware changes. (Engine and profile files are not portable and optimized for specific Nvidia hardware)

* `ORT_TENSORRT_CACHE_PATH`: Specify path for TensorRT engine and profile files if `ORT_TENSORRT_ENGINE_CACHE_ENABLE` is 1, or path for INT8 calibration table file if ORT_TENSORRT_INT8_ENABLE is 1.

* `ORT_TENSORRT_DUMP_SUBGRAPHS`: Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem. This can help debugging subgraphs, e.g. by using  `trtexec --onnx my_model.onnx` and check the outputs of the parser. 1: enabled, 0: disabled. Default value: 0.

* `ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD`: Sequentially build TensorRT engines across provider instances in multi-GPU environment. 1: enabled, 0: disabled. Default value: 0.

* `ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE`: Share execution context memory between TensorRT subgraphs. Default 0 = false, nonzero = true.

* `ORT_TENSORRT_LAYER_NORM_FP32_FALLBACK`: Force Pow + Reduce ops in layer norm to FP32. Default 0 = false, nonzero = true.

* `ORT_TENSORRT_TIMING_CACHE_ENABLE`: Enable TensorRT timing cache. Default 0 = false, nonzero = true. Check [Timing cache](#timing-cache) for details.

* `ORT_TENSORRT_FORCE_TIMING_CACHE_ENABLE`: Force the TensorRT timing cache to be used even if device profile does not match. Default 0 = false, nonzero = true.

* `ORT_TENSORRT_DETAILED_BUILD_LOG_ENABLE`: Enable detailed build step logging on TensorRT EP with timing for each engine build. Default 0 = false, nonzero = true.

* `ORT_TENSORRT_BUILD_HEURISTICS_ENABLE`: Build engine using heuristics to reduce build time. Default 0 = false, nonzero = true.

* `ORT_TENSORRT_SPARSITY_ENABLE`: Control if sparsity can be used by TRT. Default 0 = false, 1 = true. Check `--sparsity` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `ORT_TENSORRT_BUILDER_OPTIMIZATION_LEVEL`: Set the builder optimization level. WARNING: levels below 3 do not guarantee good engine performance, but greatly improve build time.  Default 3, valid range [0-5]. Check `--builderOptimizationLevel` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `ORT_TENSORRT_AUXILIARY_STREAMS`: Set maximum number of auxiliary streams per inference stream. Setting this value to 0 will lead to optimal memory usage. Default -1 = heuristics. Check `--maxAuxStreams` in `trtexec` command-line flags for [details](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags).

* `ORT_TENSORRT_TACTIC_SOURCES`: Specify the tactics to be used by adding (+) or removing (-) tactics from the default tactic sources (default = all available tactics) e.g. "-CUDNN,+CUBLAS" available keys: "CUBLAS", "CUBLAS_LT", "CUDNN" or "EDGE_MASK_CONVOLUTIONS".

* `ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS`: Specify extra TensorRT plugin library paths. ORT TRT by default supports any TRT plugins registered in TRT registry in TRT plugin library (i.e., `libnvinfer_plugin.so`). Moreover, if users want to use other TRT plugins that are not in TRT plugin library, for example, FasterTransformer has many TRT plugin implementations for different models, user can specify like this `ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS=libvit_plugin.so;libvit_int8_plugin.so`.

* `ORT_TENSORRT_PROFILE_MIN_SHAPES`, `ORT_TENSORRT_PROFILE_MAX_SHAPES` and `ORT_TENSORRT_PROFILE_OPT_SHAPES` : Build with dynamic shapes using a profile with the min/max/opt shapes provided. The format of the profile shapes is "input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,..." and these three flags should all be provided in order to enable explicit profile shapes feature. Check [Explicit shape range for dynamic shape input](#explicit-shape-range-for-dynamic-shape-input) and TRT doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles) for more details.

One can override default values by setting environment variables. e.g. on Linux:

```bash
# Override default max workspace size to 2GB
export ORT_TENSORRT_MAX_WORKSPACE_SIZE=2147483648

# Override default maximum number of iterations to 10
export ORT_TENSORRT_MAX_PARTITION_ITERATIONS=10

# Override default minimum subgraph node size to 5
export ORT_TENSORRT_MIN_SUBGRAPH_SIZE=5

# Enable FP16 mode in TensorRT
export ORT_TENSORRT_FP16_ENABLE=1

# Enable INT8 mode in TensorRT
export ORT_TENSORRT_INT8_ENABLE=1

# Use native TensorRT calibration table
export ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE=1

# Enable TensorRT engine caching
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
# Please Note warning above. This feature is experimental.
# Engine cache files must be invalidated if there are any changes to the model, ORT version, TensorRT version or if the underlying hardware changes. Engine files are not portable across devices.

# Specify TensorRT cache path
export ORT_TENSORRT_CACHE_PATH="/path/to/cache"

# Dump out subgraphs to run on TensorRT
export ORT_TENSORRT_DUMP_SUBGRAPHS=1

# Enable context memory sharing between TensorRT subgraphs. Default 0 = false, nonzero = true
export ORT_TENSORRT_CONTEXT_MEMORY_SHARING_ENABLE=1
```

</details>

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](./../performance/tune-performance/index.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e tensorrt`. Check below for sample.

### Shape Inference for TensorRT Subgraphs
If some operators in the model are not supported by TensorRT, ONNX Runtime will partition the graph and only send supported subgraphs to TensorRT execution provider. Because TensorRT requires that all inputs of the subgraphs have shape specified, ONNX Runtime will throw error if there is no input shape info. In this case please run shape inference for the entire model first by running script [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py) (Check below for sample).

### TensorRT Plugins Support
ORT TRT can leverage the TRT plugins which come with TRT plugin library in official release. To use TRT plugins, firstly users need to create the custom node (a one-to-one mapping to TRT plugin) with a registered plugin name and `trt.plugins` domain in the ONNX model. So, ORT TRT can recognize this custom node and pass the node together with the subgraph to TRT. Please see following python example to create a new custom node in the ONNX model:

Click below for Python API example:

<details>

```python
from onnx import TensorProto, helper

def generate_model(model_name):
    nodes = [
        helper.make_node(
            "DisentangledAttention_TRT", # The registered name is from https://github.com/NVIDIA/TensorRT/blob/main/plugin/disentangledAttentionPlugin/disentangledAttentionPlugin.cpp#L36
            ["input1", "input2", "input3"],
            ["output"],
            "DisentangledAttention_TRT",
            domain="trt.plugins", # The domain has to be "trt.plugins"
            factor=0.123,
            span=128,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "trt_plugin_custom_op",
        [  # input
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, [12, 256, 256]),
            helper.make_tensor_value_info("input2", TensorProto.FLOAT, [12, 256, 256]),
            helper.make_tensor_value_info("input3", TensorProto.FLOAT, [12, 256, 256]),
        ],
        [  # output
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [12, 256, 256]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)
```

</details>

Note: If users want to use TRT plugins that are not in the TRT plugin library in official release, please see the ORT TRT provider option `trt_extra_plugin_lib_paths` for more details.

### Timing cache
Enabling `trt_timing_cache_enable` will enable ORT TRT to use TensorRT timing cache to accelerate engine build time on a device with the same compute capability. This will work across models as it simply stores kernel latencies for specific configurations and cubins (TRT 9.0+). Those files are usually very small (only a few KB or MB) which makes them very easy to ship with an application to accelerate the build time on the user end.

_Note:_ A timing cache can be used across one [GPU compute capability](https://developer.nvidia.com/cuda-gpus) similar to an engine. Nonetheless the preferred way is to use one per GPU model, but practice shows that sharing across one compute capability works well in most cases.

The following examples shows build time reduction with timing cache:

|Model | no Cache | with Cache|
| ------------- | ------------- | ------------- |
|efficientnet-lite4-11 | 34.6 s | 7.7 s|
|yolov4 | 108.62 s | 9.4 s|

Click below for Python example:

<details>

```python
import onnxruntime as ort

ort.set_default_logger_severity(0) # Turn on verbose mode for ORT TRT
sess_options = ort.SessionOptions()

trt_ep_options = {
    "trt_timing_cache_enable": True,
}

sess = ort.InferenceSession(
    "my_model.onnx",
    providers=[
        ("TensorrtExecutionProvider", trt_ep_options),
        "CUDAExecutionProvider",
    ],
)

# Once inference session initialization is done (assume no dynamic shape input, otherwise you must wait until inference run is done)
# you can find timing cache is saved in the 'trt_engine_cache_path' directory, e.g., TensorrtExecutionProvider_cache_cc75.timing, please note
# that the name contains information of compute capability.

sess.run(
    None,
    {"input_ids": np.zeros((1, 77), dtype=np.int32)}
)
```

</details>

### Explicit shape range for dynamic shape input

ORT TRT lets you explicitly specify min/max/opt shapes for each dynamic shape input through three provider options, `trt_profile_min_shapes`, `trt_profile_max_shapes` and `trt_profile_opt_shapes`. If these three provider options are not specified
and model has dynamic shape input, ORT TRT will determine the min/max/opt shapes for the dynamic shape input based on incoming input tensor. The min/max/opt shapes are required for TRT optimization profile (An optimization profile describes a range of dimensions for each TRT network input and the dimensions that the auto-tuner will use for optimization. When using runtime dimensions, you must create at least one optimization profile at build time.)

To use the engine cache built with optimization profiles specified by explicit shape ranges, user still needs to provide those three provider options as well as engine cache enable flag.
ORT TRT will firstly compare the shape ranges of those three provider options with the shape ranges saved in the .profile file, and then rebuild the engine if the shape ranges don't match.

Click below for Python example:

<details>

```python
import onnxruntime as ort

ort.set_default_logger_severity(0) # Turn on verbose mode for ORT TRT
sess_options = ort.SessionOptions()

trt_ep_options = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_profile_min_shapes": "sample:2x4x64x64,encoder_hidden_states:2x77x768",
    "trt_profile_max_shapes": "sample:32x4x64x64,encoder_hidden_states:32x77x768",
    "trt_profile_opt_shapes": "sample:2x4x64x64,encoder_hidden_states:2x77x768",
}

sess = ort.InferenceSession(
    "my_model.onnx",
    providers=[
        ("TensorrtExecutionProvider", trt_ep_options),
        "CUDAExecutionProvider",
    ],
)

batch_size = 1
unet_dim = 4
max_text_len = 77
embed_dim = 768
latent_height = 64
latent_width = 64

args = {
    "sample": np.zeros(
        (2 * batch_size, unet_dim, latent_height, latent_width), dtype=np.float32
    ),
    "timestep": np.ones((1,), dtype=np.float32),
    "encoder_hidden_states": np.zeros(
        (2 * batch_size, max_text_len, embed_dim),
        dtype=np.float32,
    ),
}
sess.run(None, args)
# you can find engine cache and profile cache are saved in the 'trt_engine_cache_path' directory, e.g.
# TensorrtExecutionProvider_TRTKernel_graph_torch_jit_1843998305741310361_0_0_fp16.engine and TensorrtExecutionProvider_TRTKernel_graph_torch_jit_1843998305741310361_0_0_fp16.profile.

```

</details>

Please note that there is a constraint of using this explicit shape range feature, i.e., all the dynamic shape inputs should be provided with corresponding min/max/opt shapes.


## Samples

This example shows how to run the Faster R-CNN model on TensorRT execution provider.

1. Download the Faster R-CNN onnx model from the ONNX model zoo [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn).

2. Infer shapes in the model by running the [shape inference script](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py)
    ```bash
    python symbolic_shape_infer.py --input /path/to/onnx/model/model.onnx --output /path/to/onnx/model/new_model.onnx --auto_merge
    ```

3. To test model with sample input and verify the output, run `onnx_test_runner` under ONNX Runtime build directory.

    > Models and test_data_set_ folder need to be stored under the same path. `onnx_test_runner` will test all models under this path.

    ```bash
    ./onnx_test_runner -e tensorrt /path/to/onnx/model/
    ```

4. To test on model performance, run `onnxruntime_perf_test` on your shape-inferred Faster-RCNN model

   > Download sample test data with model from [model zoo](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/faster-rcnn), and put test_data_set folder next to your inferred model

    ```bash
    # e.g.
    # -r: set up test repeat time
    # -e: set up execution provider
    # -i: set up params for execution provider options
    ./onnxruntime_perf_test -r 1 -e tensorrt -i "trt_fp16_enable|true" /path/to/onnx/your_inferred_model.onnx
    ```

Please see [this Notebook](https://github.com/microsoft/onnxruntime/blob/main/docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb) for an example of running a model on GPU using ONNX Runtime through Azure Machine Learning Services.
