---
title: Intel - OpenVINO™
description: Instructions to execute OpenVINO™ Execution Provider for ONNX Runtime.
parent: Execution Providers
nav_order: 3
redirect_from: /docs/reference/execution-providers/OpenVINO-ExecutionProvider
---

# OpenVINO™ Execution Provider
{: .no_toc }

Accelerate ONNX models on Intel CPUs, GPUs with Intel OpenVINO™ Execution Provider. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Pre-built packages and Docker images are published for OpenVINO™ Execution Provider for ONNX Runtime by Intel for each release.
* OpenVINO™ Execution Provider for ONNX Runtime Release page: [Latest v5.1 Release](https://github.com/intel/onnxruntime/releases)
* Python wheels Ubuntu/Windows: [onnxruntime-openvino](https://pypi.org/project/onnxruntime-openvino/)
* Docker image: [openvino/onnxruntime_ep_ubuntu20](https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu20)

## Requirements

ONNX Runtime OpenVINO™ Execution Provider is compatible with three lastest releases of OpenVINO™.

|ONNX Runtime|OpenVINO™|Notes|
|---|---|---|
|1.16.0|2023.1|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.1)|
|1.15.0|2023.0|[Details](https://github.com/intel/onnxruntime/releases/tag/v5.0.0)|
|1.14.0|2022.3|[Details](https://github.com/intel/onnxruntime/releases/tag/v4.3)|

## Build

For build instructions, please see the [BUILD page](../build/eps.md#openvino).

## Usage

**Set OpenVINO™ Environment for Python**

Please download onnxruntime-openvino python packages from PyPi.org:
```
pip install onnxruntime-openvino
```

* **Windows**

   To enable OpenVINO™ Execution Provider with ONNX Runtime on Windows it is must to set up the OpenVINO™ Environment Variables using the full installer package of OpenVINO™.
   Initialize the OpenVINO™ environment by running the setupvars script as shown below. This is a required step:

   ```
      C:\ <openvino_install_directory>\setupvars.bat
   ```

* **Linux**

   OpenVINO™ Execution Provider with Onnx Runtime on Linux, installed from PyPi.org comes with prebuilt OpenVINO™ libs and supports flag CXX11_ABI=0. So there is no need to install OpenVINO™ separately.

   But if there is need to enable CX11_ABI=1 flag of OpenVINO, build Onnx Runtime python wheel packages from source. For build instructions, please see the [BUILD page](../build/eps.md#openvino).
   OpenVINO™ Execution Provider wheels on Linux built from source will not have prebuilt  OpenVINO™ libs so we must set the OpenVINO™ Environment Variable using the full installer package of OpenVINO™:

      ```
      $ source <openvino_install_directory>/setupvars.sh
      ```

**Set OpenVINO™ Environment for C++**

For Running C++/C# ORT Samples with the OpenVINO™ Execution Provider it is must to set up the OpenVINO™ Environment Variables using the full installer package of OpenVINO™.
Initialize the OpenVINO™ environment by running the setupvars script as shown below. This is a required step:
   * For Windows run:
   ```
      C:\ <openvino_install_directory>\setupvars.bat
   ```
   * For Linux run:
   ```
      $ source <openvino_install_directory>/setupvars.sh
   ```
   **Note:** If you are using a dockerfile to use OpenVINO™ Execution Provider, sourcing OpenVINO™ won't be possible within the dockerfile. You would have to explicitly set the LD_LIBRARY_PATH to point to OpenVINO™ libraries location. Refer our [dockerfile](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.openvino).


**Set OpenVINO™ Environment for  C#**

To use csharp api for openvino execution provider create a custom nuget package. Follow the instructions [here](../build/inferencing.md#build-nuget-packages) to install prerequisites for nuget creation. Once prerequisites are installed follow the instructions to [build openvino execution provider](../build/eps.md#openvino) and add an extra flag `--build_nuget` to create nuget packages. Two nuget packages will be created Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Openvino.

## Features

### OpenCL queue throttling for GPU devices

Enables [OpenCL queue throttling](https://docs.openvino.ai/latest/groupov_runtime_ocl_gpu_prop_cpp_api.html?highlight=throttling) for GPU devices. Reduces CPU utilization when using GPUs with OpenVINO EP.

### Model caching

OpenVINO™ supports [model caching](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_caching_overview.html).

From OpenVINO™ 2023.1 version, model caching feature is supported on CPU, GPU along with kernel caching on iGPU, dGPU.

This feature enables users to save and load the blob file directly on to the hardware device target and perform inference with improved Inference Latency.

Kernel Caching on iGPU and dGPU:

This feature also allows user to save kernel caching as cl_cache files for models with dynamic input shapes. These cl_cache files can be loaded directly onto the iGPU/dGPU hardware device target and inferencing can be performed.

#### <b> Enabling Model Caching via Runtime options using c++/python API's.</b>

This flow can be enabled by setting the runtime config option 'cache_dir' specifying the path to dump and load the blobs (CPU, iGPU, dGPU) or cl_cache(iGPU, dGPU) while using the c++/python API'S.

Refer to [Configuration Options](#configuration-options) for more information about using these runtime options.

### Support for INT8 Quantized models

Int8 models are supported on CPU and GPU.

### Support for Weights saved in external files

OpenVINO™ Execution Provider now  supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations.

See the [OpenVINO™ ONNX Support documentation](https://docs.openvino.ai/latest/classov_1_1Core.html).

Converting and Saving an ONNX Model to External Data:
Use the ONNX API's.[documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md#converting-and-saving-an-onnx-model-to-external-data).

Example:

```python
import onnx
onnx_model = onnx.load("model.onnx") # Your model in memory as ModelProto
onnx.save_model(onnx_model, 'saved_model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='data/weights_data', size_threshold=1024, convert_attribute=False)
```

Note:
1. In the above script, model.onnx is loaded and then gets saved into a file called 'saved_model.onnx' which won't have the weights but this new onnx model now will have the relative path to where the weights file is located. The weights file 'weights_data' will now contain the weights of the model and the weights from the original model gets saved at /data/weights_data.

2. Now, you can use this 'saved_model.onnx' file to infer using your sample. But remember, the weights file location can't be changed. The weights have to be present at /data/weights_data

3. Install the latest ONNX Python package using pip to run these ONNX Python API's successfully.

### Support for IO Buffer Optimization

To enable IO Buffer Optimization we have to set OPENCL_LIBS, OPENCL_INCS environment variables before build. For IO Buffer Optimization, the model must be fully supported on OpenVINO™ and we must provide in the remote context cl_context void pointer as C++ Configuration Option. We can provide cl::Buffer address as Input using GPU Memory Allocator for input and output.

Example:
```bash
//Set up a remote context
cl::Context _context;
.....
// Set the context through openvino options
std::unordered_map<std::string, std::string> ov_options;
ov_options[context] = std::to_string((unsigned long long)(void *) _context.get());
.....
//Define the Memory area
Ort::MemoryInfo info_gpu("OpenVINO_GPU", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
//Create a shared buffer , fill in with data
cl::Buffer shared_buffer(_context, CL_MEM_READ_WRITE, imgSize, NULL, &err);
....
//Cast it to void*, and wrap it as device pointer for Ort::Value
void *shared_buffer_void = static_cast<void *>(&shared_buffer);
Ort::Value inputTensors = Ort::Value::CreateTensor(
        info_gpu, shared_buffer_void, imgSize, inputDims.data(),
        inputDims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
```

### Multi-threading for OpenVINO™ Execution Provider

OpenVINO™ Execution Provider for ONNX Runtime enables thread-safe deep learning inference

### Multi streams for OpenVINO™ Execution Provider
OpenVINO™ Execution Provider for ONNX Runtime allows multiple stream execution for difference performance requirements part of API 2.0

### Auto-Device Execution for OpenVINO EP

Use `AUTO:<device 1><device 2>..` as the device name to delegate selection of an actual accelerator to OpenVINO™. Auto-device internally recognizes and selects devices from CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristic of CNN models, for example, precisions. Then Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in full system.

For more information on Auto-Device plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Auto Device Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_AUTO.html).

### Heterogeneous Execution for OpenVINO™ Execution Provider

The heterogeneous execution enables computing for inference on one network on several devices. Purposes to execute networks in heterogeneous mode:

* To utilize accelerator's power and calculate the heaviest parts of the network on the accelerator and execute unsupported layers on fallback devices like the CPU to utilize all available hardware more efficiently during one inference.

For more information on Heterogeneous plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Heterogeneous Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Hetero_execution.html).

### Multi-Device Execution for OpenVINO EP

Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. Potential gains are as follows:

* Improved throughput that multiple devices can deliver (compared to single-device execution)
* More consistent performance, since the devices can now share the inference burden (so that if one device is becoming too busy, another device can take more of the load)

For more information on Multi-Device plugin of OpenVINO™, please refer to the
[Intel OpenVINO™ Multi Device Plugin](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Running_on_multiple_devices.html).


## Configuration Options

OpenVINO™ Execution Provider can be configured with certain options at runtime that control the behavior of the EP. These options can be set as key-value pairs as below:-

### Python API
Key-Value pairs for config options can be set using InferenceSession API as follow:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, providers=['OpenVINOExecutionProvider'], provider_options=[{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that the releases from (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution providers other than the default CPU provider (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.*

### C/C++ API 2.0 
The session configuration options are passed to SessionOptionsAppendExecutionProvider API as shown in an example below for GPU device type:

```
std::unordered_map<std::string, std::string> options;
options[device_type] = "GPU_FP32";
options[device_id] = "";
options[num_of_threads] = "8";
options[num_streams] = "8";
options[cache_dir] = "";
options[context] = "0x123456ff";
options[enable_opencl_throttling] = "false";
session_options.AppendExecutionProvider("OpenVINO", options);
```

### C/C++ Legacy API
The session configuration options are passed to SessionOptionsAppendExecutionProvider_OpenVINO() API as shown in an example below for GPU device type:

```
OrtOpenVINOProviderOptions options;
options.device_type = "GPU_FP32";
options.device_id = "";
options.num_of_threads = 8;
options.cache_dir = "";
options.context = 0x123456ff;
options.enable_opencl_throttling = false;
SessionOptions.AppendExecutionProvider_OpenVINO(session_options, &options);
```

### Onnxruntime Graph level Optimization
OpenVINO™ backend performs hardware, dependent as well as independent optimizations on the graph to infer it on the target hardware with best possible performance. In most cases it has been observed that passing the ONNX input graph as is without explicit optimizations would lead to best possible optimizations at kernel level by OpenVINO™. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime for OpenVINO™ Execution Provider. This can be done using SessionOptions() as shown below:-

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

## Summary of options

The following table lists all the available configuration options for API 2.0 and the Key-Value pairs to set them:

| **Key** | **Key type** | **Allowable Values** | **Value type** | **Description** |
| --- | --- | --- | --- | --- |
| device_type | string | CPU_FP32, CPU_FP16, GPU_FP32, GPU_FP16, GPU.0_FP32, GPU.1_FP32, GPU.0_FP16, GPU.1_FP16 based on the avaialable GPUs, Any valid Hetero combination, Any valid Multi or Auto devices combination | string | Overrides the accelerator hardware type and precision with these values at runtime. If this option is not explicitly set, default hardware and precision specified during build time is used. |Overrides the accelerator hardware type and precision with these values at runtime. If this option is not explicitly set, default hardware and precision specified during build time is used. |
| device_id   | string | Any valid OpenVINO device ID | string | Selects a particular hardware device for inference. The list of valid OpenVINO device ID's available on a platform can be obtained either by Python API (`onnxruntime.capi._pybind_state.get_available_openvino_device_ids()`) or by [OpenVINO C/C++ API](https://docs.openvino.ai/latest/classInferenceEngine_1_1Core.html). If this option is not explicitly set, an arbitrary free device will be automatically selected by OpenVINO runtime.|
| num_of_threads | string | Any unsigned positive number other than 0 | size_t | Overrides the accelerator default value of number of threads with this value at runtime. If this option is not explicitly set, default value of 8 during build time will be used for inference. |
| num_streams | string | Any unsigned positive number other than 0 | size_t | Overrides the accelerator default streams with this value at runtime. If this option is not explicitly set, default value of 1, performance for latency is used during build time will be used for inference. |
| cache_dir | string | Any valid string path on the hardware target | string | Explicitly specify the path to save and load the blobs enabling model caching feature.|
| context | string | OpenCL Context | void* | This option is only available when OpenVINO EP is built with OpenCL flags enabled. It takes in the remote context i.e the cl_context address as a void pointer.|
| enable_opencl_throttling | string | True/False | boolean | This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU). |


Valid Hetero or Multi or Auto Device combinations:
HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...
The <DEVICE_TYPE> can be any of these devices from this list ['CPU','GPU']

A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO or Multi-Device Build.

Example:
HETERO:GPU,CPU  AUTO:GPU,CPU  MULTI:GPU,CPU


## Support Coverage

**ONNX Layers supported using OpenVINO**

The table below shows the ONNX layers supported and validated using OpenVINO™ Execution Provider.The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. Intel Discrete Graphics

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
| Greater | Yes | Yes | Yes |
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

Below topologies from ONNX open model zoo are fully supported on OpenVINO™ Execution Provider and many more are supported through sub-graph partitioning

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

**Note:** We have added support for INT8 models, quantized with Neural Network Compression Framework (NNCF). To know more about NNCF refer [here](https://github.com/openvinotoolkit/nncf).

## OpenVINO™ Execution Provider Samples Tutorials

In order to showcase what you can do with the OpenVINO™ Execution Provider for ONNX Runtime, we have created a few samples that shows how you can get that performance boost you’re looking for with just one additional line of code.

### Python API
[Object detection with tinyYOLOv2 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/tiny_yolo_v2_object_detection)

[Object detection with YOLOv4 in Python](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/yolov4_object_detection)

### C/C++ API
[Image classification with Squeezenet in CPP](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/OpenVINO_EP)

### Csharp API
[Object detection with YOLOv3 in C#](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_sharp/OpenVINO_EP/yolov3_object_detection)

## Blogs/Tutorials

### Overview of OpenVINO Execution Provider for ONNX Runtime
[OpenVINO Execution Provider](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/faster-inferencing-with-one-line-of-code.html)

### Tutorial on how to use OpenVINO™ Execution Provider for ONNX Runtime Docker Containers
[Docker Containers](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-docker-container.html)

### Tutorial on how to use OpenVINO™ Execution Provider for ONNX Runtime python wheel packages
[Python Pip Wheel Packages](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/openvino-execution-provider-for-onnx-runtime.html)
