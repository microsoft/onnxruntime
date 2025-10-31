---
title: Intel - OpenVINO™
description: Instructions to execute OpenVINO™ Execution Provider for ONNX Runtime.
parent: Execution Providers
nav_order: 3
redirect_from: /docs/reference/execution-providers/OpenVINO-ExecutionProvider
---

# OpenVINO™ Execution Provider
{: .no_toc }

Accelerate ONNX models on Intel CPUs, GPUs, NPU with Intel OpenVINO™ Execution Provider. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Intel publishes pre-built OpenVINO™ Execution Provider packages for ONNX Runtime with each release.
* OpenVINO™ Execution Provider for ONNX Runtime Release page: [Latest v5.8 Release](https://github.com/intel/onnxruntime/releases)
* Python wheels Ubuntu/Windows: [onnxruntime-openvino](https://pypi.org/project/onnxruntime-openvino/)

## Requirements


ONNX Runtime OpenVINO™ Execution Provider is compatible with three latest releases of OpenVINO™.

|ONNX Runtime|OpenVINO™|Notes|
|---|---|---| 
|1.23.0|2025.3|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.8)|
|1.22.0|2025.1|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.7)|
|1.21.0|2025.0|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.6)|

## Build

For build instructions, refer [BUILD page](../build/eps.md#openvino).

## Usage

**Python Package Installation**

For Python users, install the onnxruntime-openvino package:
```
pip install onnxruntime-openvino
```

**Set OpenVINO™ Environment Variables**

To use OpenVINO™ Execution Provider with any programming language (Python, C++, C#), you must set up the OpenVINO™ Environment Variables using the full installer package of OpenVINO™.

* **Windows**
```
C:\ <openvino_install_directory>\setupvars.bat
```
* **Linux**
```
$ source <openvino_install_directory>/setupvars.sh
```



**Set OpenVINO™ Environment for  C#**

To use csharp api for openvino execution provider create a custom nuget package. Follow the instructions [here](../build/inferencing.md#build-nuget-packages) to install prerequisites for nuget creation. Once prerequisites are installed follow the instructions to [build openvino execution provider](../build/eps.md#openvino) and add an extra flag `--build_nuget` to create nuget packages. Two nuget packages will be created Microsoft.ML.OnnxRuntime.Managed and Intel.ML.OnnxRuntime.Openvino.


## Configuration Options

Runtime parameters set during OpenVINO Execution Provider initialization to control the inference flow.


| **Key** | **Type** | **Allowable Values** | **Value Type** | **Description** |
|---------|----------|---------------------|----------------|-----------------|
| [**device_type**](#device_type) | string | CPU, NPU, GPU, GPU.0, GPU.1, HETERO, MULTI, AUTO | string | Specify intel target H/W device |
| [**precision**](#precision) | string | FP32, FP16, ACCURACY | string | Set inference precision level |
| [**num_of_threads**](#num_of_threads--num_streams) | string | Any positive integer > 0 | size_t | Control number of inference threads |
| [**num_streams**](#num_of_threads--num_streams) | string | Any positive integer > 0 | size_t | Set parallel execution streams for throughput |
| [**cache_dir**](#cache_dir) | string | Valid filesystem path | string | Enable openvino model caching for improved latency  |
| [**load_config**](#load_config) | string | JSON file path | string | Load and set custom/HW specific OpenVINO properties from JSON |
| [**enable_qdq_optimizer**](#enable_qdq_optimizer) | string | True/False | boolean | Enable QDQ optimization for NPU |
| [**disable_dynamic_shapes**](#disable_dynamic_shapes) | string | True/False | boolean | Convert dynamic models to static shapes |
| [**reshape_input**](#reshape_input) | string | input_name[shape_bounds] | string | Specify upper and lower bound for dynamic shaped inputs for improved performance with NPU |
| [**layout**](#layout) | string | input_name[layout_format] | string | Specify input/output tensor layout format |

**Deprecation Notice**

The following provider options are **deprecated** and should be migrated to `load_config` for better compatibility with future releases.

| Deprecated Provider Option | `load_config` Equivalent | Recommended Migration |
|---------------------------|------------------------|----------------------|
| `precision="FP16"` | `INFERENCE_PRECISION_HINT` | `{"GPU": {"INFERENCE_PRECISION_HINT": "f16"}}` |
| `precision="FP32"` | `INFERENCE_PRECISION_HINT` | `{"GPU": {"INFERENCE_PRECISION_HINT": "f32"}}` |
| `precision="ACCURACY"` | `EXECUTION_MODE_HINT` | `{"GPU": {"EXECUTION_MODE_HINT": "ACCURACY"}}` |
| `num_of_threads=8` | `INFERENCE_NUM_THREADS` | `{"CPU": {"INFERENCE_NUM_THREADS": "8"}}` |
| `num_streams=4` | `NUM_STREAMS` | `{"GPU": {"NUM_STREAMS": "4"}}` |

Refer to [Examples](#examples) for usage.

## Configuration Descriptions

### `device_type`

Specify the target hardware device for compilation and inference execution. The OpenVINO Execution Provider supports the following devices for deep learning model execution: **CPU**, **GPU**, and **NPU**. Configuration supports both single device and multi-device setups, enabling:
- Automatic device selection
- Heterogeneous inference across devices
- Multi-device parallel execution

**Supported Devices:**

- `CPU` — Intel CPU
- `GPU` — Intel integrated GPU or discrete GPU
- `GPU.0`, `GPU.1` — Specific GPU when multiple GPUs are available
- `NPU` — Intel Neural Processing Unit

**Multi-Device Configurations:**

OpenVINO offers the option of running inference with the following inference modes:

- `AUTO:<device1>,<device2>...` — Automatic Device Selection
- `HETERO:<device1>,<device2>...` — Heterogeneous Inference  
- `MULTI:<device1>,<device2>...` — Multi-Device Execution

Minimum **two devices** required for multi-device configurations.

**Examples:**
- `AUTO:GPU,NPU,CPU`
- `HETERO:GPU,CPU`
- `MULTI:GPU,CPU`

**Automatic Device Selection**

Automatically selects the best device available for the given task. It offers many additional options and optimizations, including inference on multiple devices at the same time. AUTO internally recognizes CPU, integrated GPU, discrete Intel GPUs, and NPU, then assigns inference requests to the best-suited device.

**Heterogeneous Inference**

Enables splitting inference among several devices automatically. If one device doesn't support certain operations, HETERO distributes the workload across multiple devices, utilizing accelerator power for heavy operations while falling back to CPU for unsupported layers.

**Multi-Device Execution**

Runs the same model on multiple devices in parallel to improve device utilization. MULTI automatically groups inference requests to improve throughput and performance consistency via load distribution.

**Note:**   Deprecated options `CPU_FP32`, `GPU_FP32`, `GPU_FP16`, `NPU_FP16` are no longer supported. Use `device_type` and `precision` separately.

---

### `precision`
**DEPRECATED:** This option is deprecated and can be set via `load_config` using the `INFERENCE_PRECISION_HINT` property.
- Controls numerical precision during inference, balancing **performance** and **accuracy**.

**Precision Support on Devices:**

- **CPU:** `FP32`
- **GPU:** `FP32`, `FP16`, `ACCURACY`
- **NPU:** `FP16`

**ACCURACY Mode**

- Maintains original model precision without conversion, ensuring maximum accuracy.

**Note 1:** `FP16` generally provides ~2x better performance on GPU/NPU with minimal accuracy loss.



---
### `num_of_threads` & `num_streams`

**DEPRECATED:** These options are deprecated and can be set via `load_config` using the `INFERENCE_NUM_THREADS` and `NUM_STREAMS` properties respectively.

**Multi-Threading**

- Controls the number of inference threads for CPU execution (default: `8`). OpenVINO EP provides thread-safe inference across all devices.

**Multi-Stream Execution**

Manages parallel inference streams for throughput optimization (default: `1` for latency-focused execution).

- **Multiple streams:** Higher throughput for batch workloads
- **Single stream:** Lower latency for real-time applications


---

### `cache_dir`

**DEPRECATED:** This option is deprecated and can be set via `load_config` using the `CACHE_DIR` property.

Enables model caching to significantly reduce subsequent load times. Supports CPU, NPU, and GPU devices with kernel caching on iGPU/dGPU.

**Benefits**
- Saves compiled models and `cl_cache` files for dynamic shapes
- Eliminates recompilation overhead on subsequent runs
- Particularly useful for complex models and frequent application restarts


---

### `load_config`

**Recommended Configuration Method** for setting OpenVINO runtime properties. Provides direct access to OpenVINO properties through a JSON configuration file during runtime.

#### Overview

`load_config` enables fine-grained control over OpenVINO inference behavior by loading properties from a JSON file. This is the **preferred method** for configuring advanced OpenVINO features, offering:

- Direct access to OpenVINO runtime properties
- Device-specific configuration
- Better compatibility with future OpenVINO releases
- No property name translation required

#### JSON Configuration Format
```json
{
  "DEVICE_NAME": {
    "PROPERTY_KEY": "value"
  }
}
```

**Supported Device Names:**
- `"CPU"` - Intel CPU
- `"GPU"` - Intel integrated/discrete GPU
- `"NPU"` - Intel Neural Processing Unit
- `"AUTO"` - Automatic device selection


#### Popular OpenVINO Properties

The following properties are commonly used for optimizing inference performance. For complete property definitions and all possible values, refer to the [OpenVINO properties](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/properties.hpp) header file.
##### Performance & Execution Hints

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `PERFORMANCE_HINT` | `"LATENCY"`, `"THROUGHPUT"` | High-level performance optimization goal |
| `EXECUTION_MODE_HINT` | `"ACCURACY"`, `"PERFORMANCE"` | Accuracy vs performance trade-off |
| `INFERENCE_PRECISION_HINT` | `"f32"`, `"f16"`, `"bf16"` | Explicit inference precision |


**PERFORMANCE_HINT:**
- `"LATENCY"`: Optimizes for low latency
- `"THROUGHPUT"`: Optimizes for high throughput

**EXECUTION_MODE_HINT:**
- `"ACCURACY"`: Maintains model precision, dynamic precision selection
- `"PERFORMANCE"`: Optimizes for speed, may use lower precision

**INFERENCE_PRECISION_HINT:**
- `"f16"`: FP16 precision 
- `"f32"`: FP32 precision - highest accuracy
- `"bf16"`: BF16 precision - balance between f16 and f32

**Important:** Use either `EXECUTION_MODE_HINT` OR `INFERENCE_PRECISION_HINT`, not both. These properties control similar behavior and should not be combined.

**Note:** CPU accepts `"f16"` hint in configuration but will upscale to FP32 during execution, as CPU only supports FP32 precision natively.


##### Threading & Streams

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `NUM_STREAMS` | Positive integer (e.g., `"1"`, `"4"`, `"8"`) | Number of parallel execution streams |
| `INFERENCE_NUM_THREADS` | Integer | Maximum number of inference threads |
| `COMPILATION_NUM_THREADS` | Integer  | Maximum number of compilation threads | 

**NUM_STREAMS:**
- Controls parallel execution streams for throughput optimization
- Higher values increase throughput for batch processing
- Lower values optimize latency for real-time inference

**INFERENCE_NUM_THREADS:**
- Controls CPU thread count for inference execution
- Explicit value: Fixed thread count (e.g., `"4"` limits to 4 threads)

##### Caching Properties

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `CACHE_DIR` | File path string | Model cache directory |
| `CACHE_MODE` | `"OPTIMIZE_SIZE"`, `"OPTIMIZE_SPEED"` | Cache optimization strategy |

**CACHE_MODE:**
- `"OPTIMIZE_SPEED"`: Faster cache creation, larger cache files
- `"OPTIMIZE_SIZE"`: Slower cache creation, smaller cache files
##### Logging Properties

| Property | Valid Values | Description | 
|----------|-------------|-------------|
| `LOG_LEVEL` | `"LOG_NONE"`, `"LOG_ERROR"`, `"LOG_WARNING"`, `"LOG_INFO"`, `"LOG_DEBUG"`, `"LOG_TRACE"` | Logging verbosity level | 

**Note:** `LOG_LEVEL` is not supported on GPU devices.

##### AUTO Device Properties

| Property | Valid Values | Description |
|----------|-------------|-------------|
| `ENABLE_STARTUP_FALLBACK` | `"YES"`, `"NO"` | Enable device fallback during model loading |
| `ENABLE_RUNTIME_FALLBACK` | `"YES"`, `"NO"` | Enable device fallback during inference runtime |
| `DEVICE_PROPERTIES` | Nested JSON string | Device-specific property configuration |

**DEVICE_PROPERTIES Syntax:**

Used to configure properties for individual devices when using AUTO mode.
```json
{
  "AUTO": {
    "DEVICE_PROPERTIES": "{CPU:{PROPERTY:value},GPU:{PROPERTY:value}}"
  }
}
```

#### Property Reference Documentation

For complete property definitions and advanced options, refer to the official OpenVINO properties header:

**[OpenVINO Runtime Properties](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/properties.hpp)**

Property keys used in `load_config` JSON must match the string literal defined in the properties header file.





---
  

### `enable_qdq_optimizer`

**DEPRECATED:** This option is deprecated and can be set via `load_config` using the `NPU_QDQ_OPTIMIZATION` property.

NPU-specific optimization for Quantize-Dequantize (QDQ) operations in the inference graph. This optimizer enhances ORT quantized models by:

- Retaining QDQ operations only for supported operators
- Improving inference performance on NPU devices
- Maintaining model accuracy while optimizing execution

---

### `disable_dynamic_shapes` 

**Dynamic Shape Management**

- Handles models with variable input dimensions. 
- Provides the option to convert dynamic shapes to static shapes when beneficial for performance optimization.

---

### `reshape_input`

**NPU Shape Bounds Configuration**

- Use `reshape_input` to explicitly set dynamic shape bounds for NPU devices.

**Format:**
- Range bounds: `input_name[lower..upper]`
- Fixed shape: `input_name[fixed_shape]`

This configuration is required for optimal NPU memory allocation and management.

---

### `model_priority`

**DEPRECATED:** This option is deprecated and can be set via `load_config` using the `MODEL_PRIORITY` property.

Configures resource allocation priority for multi-model deployment scenarios.

**Priority Levels:**

| Level | Description |
|-------|-------------|
| **HIGH** | Maximum resource allocation for critical models |
| **MEDIUM** | Balanced resource sharing across models |
| **LOW** | Minimal allocation, yields resources to higher priority models |
| **DEFAULT** | System-determined priority based on workload |


---

### `layout`

- Provides explicit control over tensor memory layout for performance optimization. 
- Helps OpenVINO optimize memory access patterns and tensor operations.

**Layout Characters:**

- **N:** Batch dimension
- **C:** Channel dimension
- **H:** Height dimension
- **W:** Width dimension
- **D:** Depth dimension
- **T:** Time dimension
- **?:** Unknown/dynamic dimension

**Format:**

`input_name[LAYOUT],output_name[LAYOUT]`

**Example:**

`input_image[NCHW],output_tensor[NC]`

---

## Examples

### Python

#### Using load_config with JSON file
```python
import onnxruntime as ort
import json

# Create config file
config = {
    "AUTO": {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "PERF_COUNT": "NO",
        "DEVICE_PROPERTIES": "{CPU:{INFERENCE_PRECISION_HINT:f32,NUM_STREAMS:3},GPU:{INFERENCE_PRECISION_HINT:f32,NUM_STREAMS:5}}"
    }
}

with open("ov_config.json", "w") as f:
    json.dump(config, f)

# Use config with session
options = {"device_type": "AUTO", "load_config": "ov_config.json"}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])
```

#### Using load_config for CPU
```python
import onnxruntime as ort
import json

# Create CPU config
config = {
    "CPU": {
        "INFERENCE_PRECISION_HINT": "f32",
        "NUM_STREAMS": "3",
        "INFERENCE_NUM_THREADS": "8"
    }
}

with open("cpu_config.json", "w") as f:
    json.dump(config, f)

options = {"device_type": "CPU", "load_config": "cpu_config.json"}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])
```

#### Using load_config for GPU
```python
import onnxruntime as ort
import json

# Create GPU config with caching
config = {
    "GPU": {
        "INFERENCE_PRECISION_HINT": "f16",
        "CACHE_DIR": "./model_cache",
        "PERFORMANCE_HINT": "LATENCY"
    }
}

with open("gpu_config.json", "w") as f:
    json.dump(config, f)

options = {"device_type": "GPU", "load_config": "gpu_config.json"}
session = ort.InferenceSession("model.onnx", 
                                providers=[("OpenVINOExecutionProvider", options)])
```


--- 
### Python API
Key-Value pairs for config options can be set using InferenceSession API as follow:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, providers=['OpenVINOExecutionProvider'], provider_options=[{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that the releases from (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution providers other than the default CPU provider (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.*

--- 
### C/C++ API 2.0 
The session configuration options are passed to SessionOptionsAppendExecutionProvider API as shown in an example below for GPU device type:

```
std::unordered_map<std::string, std::string> options;
options[device_type] = "GPU";
options[precision] = "FP32";
options[num_of_threads] = "8";
options[num_streams] = "8";
options[cache_dir] = "";
options[context] = "0x123456ff";
options[enable_qdq_optimizer] = "True";
options[load_config] = "config_path.json";
session_options.AppendExecutionProvider_OpenVINO_V2(options);
```
---
### C/C++ Legacy API 
Note: This API is no longer officially supported. Users are requested to move to V2 API. 

The session configuration options are passed to SessionOptionsAppendExecutionProvider_OpenVINO() API as shown in an example below for GPU device type:

```
OrtOpenVINOProviderOptions options;
options.device_type = "CPU";
options.num_of_threads = 8;
options.cache_dir = "";
options.context = 0x123456ff;
options.enable_opencl_throttling = false;
SessionOptions.AppendExecutionProvider_OpenVINO(session_options, &options);
```
---

### Onnxruntime Graph level Optimization
OpenVINO™ backend performs hardware, dependent as well as independent optimizations on the graph to infer it on the target hardware with best possible performance. In most cases it has been observed that passing the ONNX input graph as it is without explicit optimizations would lead to best possible optimizations at kernel level by OpenVINO™. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime for OpenVINO™ Execution Provider. This can be done using SessionOptions() as shown below:-

* #### Python API
   ```
   options = onnxruntime.SessionOptions()
   options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
   sess = onnxruntime.InferenceSession(<path_to_model_file>, options)
   ```

* #### C/C++ API
   ```
   SessionOptions::SetGraphOptimizationLevel(ORT_DISABLE_ALL);
   ```
---
## Support Coverage

**ONNX Layers supported using OpenVINO**

The table below shows the ONNX layers supported and validated using OpenVINO™ Execution Provider.The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. Intel Discrete Graphics. For NPU if an op is not supported we fallback to CPU. 

| **ONNX Layers** | **CPU** | **GPU** |
| --- | --- | --- |
| Abs | Yes | Yes |
| Acos | Yes | Yes |
| Acosh | Yes | Yes |
| Add | Yes | Yes |
| And | Yes | Yes |
| ArgMax | Yes | Yes |
| ArgMin | Yes | Yes |
| Asin | Yes | Yes |
| Asinh | Yes | Yes |
| Atan | Yes | Yes |
| Atanh | Yes | Yes |
| AveragePool | Yes | Yes |
| BatchNormalization | Yes | Yes |
| BitShift | Yes | No |
| Ceil | Yes | Yes |
| Celu | Yes | Yes |
| Cast | Yes | Yes |
| Clip | Yes | Yes |
| Concat | Yes | Yes |
| Constant | Yes | Yes |
| ConstantOfShape | Yes | Yes |
| Conv | Yes | Yes |
| ConvInteger | Yes | Yes |
| ConvTranspose | Yes | Yes |
| Cos | Yes | Yes |
| Cosh | Yes | Yes |
| CumSum | Yes | Yes |
| DepthToSpace | Yes | Yes |
| DequantizeLinear | Yes | Yes |
| Div | Yes | Yes |
| Dropout | Yes | Yes |
| Einsum | Yes | Yes |
| Elu | Yes | Yes |
| Equal | Yes | Yes |
| Erf | Yes | Yes |
| Exp | Yes | Yes |
| Expand | Yes | Yes |
| EyeLike | Yes | No |
| Flatten | Yes | Yes |
| Floor | Yes | Yes |
| Gather | Yes | Yes |
| GatherElements | No | No |
| GatherND | Yes | Yes |
| Gemm | Yes | Yes |
| GlobalAveragePool | Yes | Yes |
| GlobalLpPool | Yes | Yes |
| GlobalMaxPool | Yes | Yes |
| Greater | Yes | Yes |
| GreaterOrEqual | Yes | Yes |
| GridSample | Yes | No |
| HardMax | Yes | Yes |
| HardSigmoid | Yes | Yes |
| Identity | Yes | Yes |
| If | Yes | Yes |
| ImageScaler | Yes | Yes |
| InstanceNormalization | Yes | Yes |
| LeakyRelu | Yes | Yes |
| Less | Yes | Yes |
| LessOrEqual | Yes | Yes |
| Log | Yes | Yes |
| LogSoftMax | Yes | Yes |
| Loop | Yes | Yes |
| LRN | Yes | Yes |
| LSTM | Yes | Yes |
| MatMul | Yes | Yes |
| MatMulInteger | Yes | No |
| Max | Yes | Yes |
| MaxPool | Yes | Yes |
| Mean | Yes | Yes |
| MeanVarianceNormalization | Yes | Yes |
| Min | Yes | Yes |
| Mod | Yes | Yes |
| Mul | Yes | Yes |
| Neg | Yes | Yes |
| NonMaxSuppression | Yes | Yes |
| NonZero | Yes | No |
| Not | Yes | Yes |
| OneHot | Yes | Yes |
| Or | Yes | Yes |
| Pad | Yes | Yes |
| Pow | Yes | Yes |
| PRelu | Yes | Yes |
| QuantizeLinear | Yes | Yes |
| QLinearMatMul | Yes | No |
| Range | Yes | Yes |
| Reciprocal | Yes | Yes |
| ReduceL1 | Yes | Yes |
| ReduceL2 | Yes | Yes |
| ReduceLogSum | Yes | Yes |
| ReduceLogSumExp | Yes | Yes |
| ReduceMax | Yes | Yes |
| ReduceMean | Yes | Yes |
| ReduceMin | Yes | Yes |
| ReduceProd | Yes | Yes |
| ReduceSum | Yes | Yes |
| ReduceSumSquare | Yes | Yes |
| Relu | Yes | Yes |
| Reshape | Yes | Yes |
| Resize | Yes | Yes |
| ReverseSequence | Yes | Yes |
| RoiAlign | Yes | Yes |
| Round | Yes | Yes |
| Scatter | Yes | Yes |
| ScatterElements | Yes | Yes |
| ScatterND | Yes | Yes |
| Selu | Yes | Yes |
| Shape | Yes | Yes |
| Shrink | Yes | Yes |
| Sigmoid | Yes | Yes |
| Sign | Yes | Yes |
| Sin | Yes | Yes |
| Sinh | Yes | No |
| SinFloat | No | No |
| Size | Yes | Yes |
| Slice | Yes | Yes |
| Softmax | Yes | Yes |
| Softplus | Yes | Yes |
| Softsign | Yes | Yes |
| SpaceToDepth | Yes | Yes |
| Split | Yes | Yes |
| Sqrt | Yes | Yes |
| Squeeze | Yes | Yes |
| Sub | Yes | Yes |
| Sum | Yes | Yes |
| Softsign | Yes | No |
| Tan | Yes | Yes |
| Tanh | Yes | Yes |
| ThresholdedRelu | Yes | Yes |
| Tile | Yes | Yes |
| TopK | Yes | Yes |
| Transpose | Yes | Yes |
| Unsqueeze | Yes | Yes |
| Upsample | Yes | Yes |
| Where | Yes | Yes |
| Xor | Yes | Yes |


### Topology Support

Below topologies from ONNX open model zoo are fully supported on OpenVINO™ Execution Provider and many more are supported through sub-graph partitioning.
For NPU if model is not supported we fallback to CPU. 

### Image Classification Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| bvlc_alexnet | Yes | Yes |
| bvlc_googlenet | Yes | Yes |
| bvlc_reference_caffenet | Yes | Yes |
| bvlc_reference_rcnn_ilsvrc13 | Yes | Yes |
| emotion ferplus | Yes | Yes |
| densenet121 | Yes | Yes |
| inception_v1 | Yes | Yes |
| inception_v2 | Yes | Yes |
| mobilenetv2 | Yes | Yes |
| resnet18v2 | Yes | Yes |
| resnet34v2 | Yes | Yes |
| resnet101v2 | Yes | Yes |
| resnet152v2 | Yes | Yes |
| resnet50 | Yes | Yes |
| resnet50v2 | Yes | Yes |
| shufflenet | Yes | Yes |
| squeezenet1.1 | Yes | Yes |
| vgg19 | Yes | Yes |
| zfnet512 | Yes | Yes |
| mxnet_arcface | Yes | Yes |


### Image Recognition Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| mnist | Yes | Yes |

### Object Detection Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| tiny_yolov2 | Yes | Yes |
| yolov3 | Yes | Yes |
| tiny_yolov3 | Yes | Yes |
| mask_rcnn | Yes | No |
| faster_rcnn | Yes | No |
| yolov4 | Yes | Yes |
| yolov5 | Yes | Yes |
| yolov7 | Yes | Yes |
| tiny_yolov7 | Yes | Yes |

### Image Manipulation Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| mosaic | Yes | Yes |
| candy | Yes | Yes |
| cgan | Yes | Yes |
| rain_princess | Yes | Yes |
| pointilism | Yes | Yes |
| udnie | Yes | Yes |

### Natural Language Processing Networks

| **MODEL NAME** | **CPU** | **GPU** |
| --- | --- | --- |
| bert-squad | Yes | Yes |
| bert-base-cased | Yes | Yes |
| bert-base-chinese | Yes | Yes |
| bert-base-japanese-char | Yes | Yes |
| bert-base-multilingual-cased | Yes | Yes |
| bert-base-uncased | Yes | Yes |
| distilbert-base-cased | Yes | Yes |
| distilbert-base-multilingual-cased | Yes | Yes |
| distilbert-base-uncased | Yes | Yes |
| distilbert-base-uncased-finetuned-sst-2-english | Yes | Yes |
| gpt2 | Yes | Yes |
| roberta-base | Yes | Yes |
| roberta-base-squad2 | Yes | Yes |
| t5-base | Yes | Yes |
| twitter-roberta-base-sentiment | Yes | Yes |
| xlm-roberta-base | Yes | Yes |

### Models Supported on NPU

| **MODEL NAME** | **NPU** |
| --- | --- |
| yolov3 | Yes |
| microsoft_resnet-50 | Yes |
| realesrgan-x4 | Yes |
| timm_inception_v4.tf_in1k | Yes |
| squeezenet1.0-qdq | Yes |
| vgg16 | Yes |
| caffenet-qdq | Yes |
| zfnet512 | Yes |
| shufflenet-v2 | Yes |
| zfnet512-qdq | Yes |
| googlenet | Yes |
| googlenet-qdq | Yes |
| caffenet | Yes |
| bvlcalexnet-qdq | Yes |
| vgg16-qdq | Yes |
| mnist | Yes |
| ResNet101-DUC | Yes |
| shufflenet-v2-qdq | Yes |
| bvlcalexnet | Yes |
| squeezenet1.0 | Yes |

**Note:** We have added support for INT8 models, quantized with Neural Network Compression Framework (NNCF). To know more about NNCF refer [here](https://github.com/openvinotoolkit/nncf).

---

# OpenVINO™ Execution Provider Samples & Tutorials

In order to showcase what you can do with the OpenVINO™ Execution Provider for ONNX Runtime, we have created a few samples that show how you can get that performance boost you're looking for with just one additional line of code.

## Samples

### Python API

- [Object detection with tinyYOLOv2 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/tiny_yolo_v2_object_detection)
- [Object detection with YOLOv4 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/yolov4_object_detection)

### C/C++ API

- [Image classification with Squeezenet in C++](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP)

### C# API

- [Object detection with YOLOv3 in C#](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_sharp/OpenVINO_EP/yolov3_object_detection)

## Blogs & Tutorials

### Overview of OpenVINO Execution Provider for ONNX Runtime

[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html) - Learn about faster inferencing with one line of code

### Python Pip Wheel Packages

[Tutorial: Using OpenVINO™ Execution Provider for ONNX Runtime Python Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)

---