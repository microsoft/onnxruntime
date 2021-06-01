---
title: TensorRT
parent: Execution Providers
grand_parent: Reference
nav_order: 2
---

# TensorRT Execution Provider
{: .no_toc }

The TensorRT execution provider in the ONNX Runtime makes use of NVIDIA's [TensorRT](https://developer.nvidia.com/tensorrt) Deep Learning inferencing engine to accelerate ONNX model in their family of GPUs. Microsoft and NVIDIA worked closely to integrate the TensorRT execution provider with ONNX Runtime.

With the TensorRT execution provider, the ONNX Runtime delivers better inferencing performance on the same hardware compared to generic GPU acceleration. 

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build

See [Build instructions](../../how-to/build/eps.md#tensorrt).

The TensorRT execution provider for ONNX Runtime is built and tested with TensorRT 7.2.2.

## Usage
### C/C++
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
int device_id = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
Ort::Session session(env, model_path, sf);
```

The C API details are [here](../api/c-api.md).

#### Shape Inference for TensorRT Subgraphs
If some operators in the model are not supported by TensorRT, ONNX Runtime will partition the graph and only send supported subgraphs to TensorRT execution provider. Because TensorRT requires that all inputs of the subgraphs have shape specified, ONNX Runtime will throw error if there is no input shape info. In this case please run shape inference for the entire model first by running script [here](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/symbolic_shape_infer.py).


### Python
When using the Python wheel from the ONNX Runtime build with TensorRT execution provider, it will be automatically prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution provider.


## Configurations
There are two ways to configure TensorRT settings, either by environment variables or by execution provider option APIs.

### Environment Variables
Following environment variables can be set for TensorRT execution provider.

* ORT_TENSORRT_MAX_WORKSPACE_SIZE: maximum workspace size for TensorRT engine. Default value: 1073741824 (1GB).

* ORT_TENSORRT_MAX_PARTITION_ITERATIONS: maximum number of iterations allowed in model partitioning for TensorRT. If target model can't be successfully partitioned when the maximum number of iterations is reached, the whole model will fall back to other execution providers such as CUDA or CPU. Default value: 1000.

* ORT_TENSORRT_MIN_SUBGRAPH_SIZE: minimum node size in a subgraph after partitioning. Subgraphs with smaller size will fall back to other execution providers. Default value: 1.

* ORT_TENSORRT_FP16_ENABLE: Enable FP16 mode in TensorRT. 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support FP16 precision.

* ORT_TENSORRT_INT8_ENABLE: Enable INT8 mode in TensorRT. 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support INT8 precision.

* ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME: Specify INT8 calibration table file name. By default the name is "INT8_calibration_table".

* ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE: Select what calibration table is used. If 1, native TensorRT generated calibration table is used; if 0, ONNXRUNTIME tool generated calibration table is used. Default value: 0.
**Note: Please copy up-to-date calibration table file to ORT_TENSORRT_CACHE_PATH before inference. Calibration table is specific to models and calibration data sets. Whenever new calibration table is generated, old file in the path should be cleaned up or be replaced.

* ORT_TENSORRT_DLA_ENABLE: Enable DLA (Deep Learning Accelerator). 1: enabled, 0: disabled. Default value: 0. Note not all Nvidia GPUs support DLA. 

* ORT_TENSORRT_DLA_CORE: Specify DLA core to execute on. Default value: 0.
 
* ORT_TENSORRT_ENGINE_CACHE_ENABLE: Enable TensorRT engine caching. The purpose of using engine caching is to save engine build time in the cases that TensorRT may take long time to optimize and build engine. Engine will be cached after it's built at the first time so that next time when inference session is created the engine can be loaded directly from cache. In order to validate that the loaded engine is usable for current inference, engine profile is also cached and loaded along with engine. If current input shapes are in the range of the engine profile, that means the loaded engine can be safely used. Otherwise if input shapes are out of range, profile cache will be updated to cover the new shape and engine will be recreated based on the new profile (and also refreshed in the engine cache). Note each engine is created for specific settings such as precision (FP32/FP16/INT8 etc), workspace, profiles etc, and specific GPUs and it's not portable, so it's essential to make sure those settings are not changing, otherwise the engines need to be rebuilt and cached again. 1: enabled, 0: disabled. Default value: 0.
**Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:**
  - Model changes (if there are any changes to the model topology, opset version etc.)
  - ORT version changes (i.e. moving from ORT version 1.4 to 1.5)
  - TensorRT version changes (i.e. moving from TensorRT 7.0 to 7.1)
  - Hardware changes. (Engine and profile files are not portable and optimized for specific Nvidia hardware)

* ORT_TENSORRT_CACHE_PATH: Specify path for TensorRT engine and profile files if ORT_TENSORRT_ENGINE_CACHE_ENABLE is 1, or path for INT8 calibration table file if ORT_TENSORRT_INT8_ENABLE is 1.

* ORT_TENSORRT_DUMP_SUBGRAPHS: Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem. This can help debugging subgraphs, e.g. by using  `trtexec --onnx my_model.onnx` and check the outputs of the parser. 1: enabled, 0: disabled. Default value: 0.

* ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD: Sequentially build TensorRT engines across provider instances in multi-GPU environment. 1: enabled, 0: disabled. Default value: 0.

One can override default values by setting environment variables ORT_TENSORRT_MAX_WORKSPACE_SIZE, ORT_TENSORRT_MAX_PARTITION_ITERATIONS, ORT_TENSORRT_MIN_SUBGRAPH_SIZE, ORT_TENSORRT_FP16_ENABLE, ORT_TENSORRT_INT8_ENABLE, ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME, ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE, ORT_TENSORRT_ENGINE_CACHE_ENABLE, ORT_TENSORRT_CACHE_PATH and ORT_TENSORRT_DUMP_SUBGRAPHS.
e.g. on Linux

#### Override default max workspace size to 2GB
export ORT_TENSORRT_MAX_WORKSPACE_SIZE=2147483648

#### Override default maximum number of iterations to 10 
export ORT_TENSORRT_MAX_PARTITION_ITERATIONS=10
        
#### Override default minimum subgraph node size to 5
export ORT_TENSORRT_MIN_SUBGRAPH_SIZE=5

#### Enable FP16 mode in TensorRT
export ORT_TENSORRT_FP16_ENABLE=1

#### Enable INT8 mode in TensorRT
export ORT_TENSORRT_INT8_ENABLE=1

#### Use native TensorRT calibration table
export ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE=1

#### Enable TensorRT engine caching
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
* Please Note warning above. This feature is experimental. Engine cache files must be invalidated if there are any changes to the model, ORT version, TensorRT version or if the
underlying hardware changes. Engine files are not portable across devices.

#### Specify TensorRT cache path
export ORT_TENSORRT_CACHE_PATH="/path/to/cache"

#### Dump out subgraphs to run on TensorRT
export ORT_TENSORRT_DUMP_SUBGRAPHS = 1

### Execution Provider Options
TensorRT configurations can also be set by execution provider option APIs. It's useful when each model and inference session have their own configurations. In this case, execution provider option settings will override any environment variable settings. All configurations should be set explicitly, otherwise default value will be taken. There are one-to-one mappings between environment variables and execution provider options shown as below,
ORT_TENSORRT_MAX_WORKSPACE_SIZE <-> trt_max_workspace_size

ORT_TENSORRT_MAX_PARTITION_ITERATIONS <-> trt_max_partition_iterations

ORT_TENSORRT_MIN_SUBGRAPH_SIZE <-> trt_min_subgraph_size

ORT_TENSORRT_FP16_ENABLE <-> trt_fp16_enable

ORT_TENSORRT_INT8_ENABLE <-> trt_int8_enable

ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME <-> trt_int8_calibration_table_name

ORT_TENSORRT_INT8_USE_NATIVE_CALIBRATION_TABLE <-> trt_int8_use_native_calibration_table

ORT_TENSORRT_DLA_ENABLE <-> trt_dla_enable

ORT_TENSORRT_DLA_CORE <-> trt_dla_core

ORT_TENSORRT_ENGINE_CACHE_ENABLE <-> trt_engine_cache_enable

ORT_TENSORRT_CACHE_PATH <-> trt_engine_cache_path

ORT_TENSORRT_DUMP_SUBGRAPHS <-> trt_dump_subgraphs

ORT_TENSORRT_FORCE_SEQUENTIAL_ENGINE_BUILD <-> trt_force_sequential_engine_build

Besides, device_id can also be set by execution provider option.

#### C++ API example
```
Ort::SessionOptions session_options;
OrtTensorRTProviderOptions trt_options{};
trt_options.device_id = 1;
trt_options.trt_max_workspace_size = 2147483648;
trt_options.trt_max_partition_iterations = 10;
trt_options.trt_min_subgraph_size = 5;
trt_options.trt_fp16_enable = 1;
trt_options.trt_int8_enable = 1;
trt_options.trt_int8_use_native_calibration_table = 1;
trt_options.trt_engine_cache_enable = 1;
trt_options.trt_engine_cache_path = "/path/to/cache"
trt_options.trt_dump_subgraphs = 1;  
session_options.AppendExecutionProvider_TensorRT(trt_options);
```

#### Python API example
```
sess_opt = ort.SessionOptions()
sess = ort.InferenceSession('model.onnx', sess_options=sess_opt)
sess.set_providers(["TensorrtExecutionProvider"],[{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable':'True'}])
```

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../../how-to/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e tensorrt` 

## Samples

This example shows how to run the Faster R-CNN model on TensorRT execution provider.

1. Download the Faster R-CNN onnx model from the ONNX model zoo [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn).

2. Infer shapes in the model by running the [shape inference script](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/symbolic_shape_infer.py)
    ```
    python symbolic_shape_infer.py --input /path/to/onnx/model/model.onnx --output /path/to/onnx/model/new_model.onnx --auto_merge
    ```

3. Replace the original model with the new model and run the 
    onnx_test_runner tool under ONNX Runtime build directory.
    ```
    ./onnx_test_runner -e tensorrt /path/to/onnx/model/
    ```

Please see [this Notebook](https://github.com/microsoft/onnxruntime/blob/master/docs/python/inference/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb) for an example of running a model on GPU using ONNX Runtime through Azure Machine Learning Services.

