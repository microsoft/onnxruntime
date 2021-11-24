---
title: OpenVINO
description: Instructions to execute ONNX Runtime with the Intel OpenVINO execution provider
parent: Execution Providers
nav_order: 10
redirect_from: /docs/reference/execution-providers/OpenVINO-ExecutionProvider
---

# OpenVINO Execution Provider
{: .no_toc }

Accelerate ONNX models on Intel CPUs, GPUs and VPUs with ONNX Runtime and the Intel OpenVINO execution provider. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install
Pre-built packages and Docker images are published for  ONNX Runtime with OpenVINO by Intel for each release.
* Python wheels: [intel/onnxruntime](https://github.com/intel/onnxruntime/releases)
* Docker image: [openvino/onnxruntime_ep_ubuntu18](https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18)

## Requirements

|ONNX Runtime|OpenVINO|Notes|
|---|---|---|
|1.10.0|2021.4.2|[Details](https://github.com/intel/onnxruntime/releases/tag/v3.4)|
|1.9.0|2021.4.1|[Details](https://github.com/intel/onnxruntime/releases/tag/v3.1)|
|1.8.1|2021.4|[Details](https://github.com/intel/onnxruntime/releases/tag/v3.0)|
|1.8.0|2021.3|[Details](https://github.com/intel/onnxruntime/releases/tag/2021.3)|

## Build
For build instructions, please see the [BUILD page](../build/eps.md#openvino).

## Usage
**C#**

To use csharp api for openvino execution provider create a custom nuget package. Follow the instructions [here](../build/inferencing.md#build-nuget-packages) to install prerequisites for nuget creation. Once prerequisites are installed follow the instructions to [build openvino](../build/eps.md#openvino) and add an extra flag `--build_nuget` to create nuget packages. Two nuget packages will be created Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Openvino. 

### Multi-threading for OpenVINO EP

OpenVINO Execution Provider enables thread-safe deep learning inference

### Heterogeneous Execution for OpenVINO EP

The heterogeneous Execution enables computing for inference on one network on several devices. Purposes to execute networks in heterogeneous mode

To utilize accelerators power and calculate heaviest parts of network on accelerator and execute not supported layers on fallback devices like CPU
To utilize all available hardware more efficiently during one inference

For more information on Heterogeneous plugin of OpenVINO, please refer to the following
[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_HETERO.html).

### Multi-Device Execution for OpenVINO EP

Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. Potential gains are as follows

Improved throughput that multiple devices can deliver (compared to single-device execution)
More consistent performance, since the devices can now share the inference burden (so that if one device is becoming too busy, another device can take more of the load)

For more information on Multi-Device plugin of OpenVINO, please refer to the following
[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html#introducing_multi_device_execution).

### Auto-Device Execution for OpenVINO EP

Use "AUTO:<device 1><device 2>.." as the device name to delegate selection of an actual accelerator to OpenVINO. With the 2021.4 release, Auto-device internally recognizes and selects devices from CPU, integrated GPU and discrete Intel GPUs (when available) depending on the device capabilities and the characteristic of CNN models, for example, precisions. Then Auto-device assigns inference requests to the selected device.

From the application point of view, this is just another device that handles all accelerators in full system.

For more information on Auto-Device plugin of OpenVINO, please refer to the following
[documentation](https://docs.openvino.ai/cn/latest/openvino_docs_IE_DG_supported_plugins_AUTO.html).

### Model caching feature for OpenVINO EP
The model caching setting enables blobs with Myriadx(VPU) and as cl_cache files with iGPU.
#### Save/Load blob capability for Myriadx(VPU)
This feature enables users to save and load the blobs directly. These pre-compiled blobs can be directly loaded on to the specific hardware device target and inferencing can be done. This feature is only supported on MyriadX(VPU) hardware device target.

#### CL Cache capability for iGPU

Starting from OpenVINO 2021.4 version, this feature is supported in OpenVINO-EP using Model caching mechanism from OpenVINO.[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Model_caching_overview.html).

This feature enables users to save and load the cl_cache files directly. These cl_cache files can be directly loaded on to igpu hardware device target and inferencing can be done. This feature is only supported on iGPU hardware device target.

There are two different methods of exercising this feature:

#### <b>option 1. Enabling via Runtime options using c++/python API's.</b>

This flow can be enabled by setting the runtime config option 'use_compiled_network' to True while using the c++/python API'S. This config option acts like a switch to on and off the feature.

The blobs are saved and loaded from a directory named 'ov_compiled_blobs' from the executable path by default. This path however can be overridden using another runtime config option 'blob_dump_path' which is used to explicitly specify the path where you would like to dump and load the blobs (VPU) or cl_cache(iGPU) files from when already using the use_compiled_network(model caching) setting.

Refer to [Configuration Options](#configuration-options) for more information about using these runtime options.

#### <b>option 2. Importing the pre-compiled blobs directly from the path set by the user.</b>

This flow enables users to import/load the pre-compiled blob directly if available readily. This option is enabled by explicitly setting the path to the blob using environment variables and setting the OV_USE_COMPILED_NETWORK flag to true.

This flow only works for MyriadX(VPU) device.

For Linux:
```
export OV_USE_COMPILED_NETWORK=1
export OV_BLOB_PATH=<path to the blob>
Example: export OV_BLOB_PATH=/home/blobs_dir/model.blob
```

For Windows:
```
set OV_USE_COMPILED_NETWORK=1
set OV_BLOB_PATH=<path to the blob>
Example: set OV_BLOB_PATH=\home\blobs_dir\model.blob
```

compile_tool:

The device specific Myriadx blobs can be generated using an offline tool called compile_tool from OpenVINO Toolkit.[documentation](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_compile_tool_README.html).

### Support for INT8 Quantized models

Starting from the OpenVINO EP 2021.4 Release, int8 models will be supported on CPU and GPU.
However, int8 support won't be available for VPU.

### Support for Weights saved in external files

Starting from the OpenVINO EP 2021.4 Release, support for external weights is added. OpenVINO™ EP now  supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations.[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_ONNX_Support.html).

Converting and Saving an ONNX Model to External Data:
Use the ONNX API's.[documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md#converting-and-saving-an-onnx-model-to-external-data).

Example:
```bash
import onnx
onnx_model = onnx.load("model.onnx") # Your model in memory as ModelProto
onnx.save_model(onnx_model, 'saved_model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='data/weights_data', size_threshold=1024, convert_attribute=False)
```

Note:
1. In the above script, model.onnx is loaded and then gets saved into a file called 'saved_model.onnx' which won't have the weights but this new onnx model now will have the relative path to where the weights file is located. The weights file 'weights_data' will now contain the weights of the model and the weights from the original model gets saved at /data/weights_data.

2. Now, you can use this 'saved_model.onnx' file to infer using your sample. But remember, the weights file location can't be changed. The weights have to be present at /data/weights_data

3. Install the latest ONNX Python package using pip to run these ONNX Python API's successfully.

### Support for IO Buffer Optimization

To enable IO Buffer Optimization we have to set OPENCL_LIBS, OPENCL_INCS environment variables before build. For IO Buffer Optimization, the model must be fully supported on
OpenVINO and we must provide in the remote context cl_context void pointer as C++ Configuration Option. We can provide cl::Buffer address as Input using GPU Memory Allocator for input and output.

Example:
```bash
//Set up a remote context
cl::Context _context;
.....
// Set the context through openvino options
OrtOpenVINOProviderOptions options;
options.context = (void *) _context.get() ;
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

### Configuration Options

OpenVINO EP can be configured with certain options at runtime that control the behavior of the EP. These options can be set as key-value pairs as below:-

### Python API
Key-Value pairs for config options can be set using InferenceSession API as follow:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, providers=['OpenVINOExecutionProvider'], provider_options=[{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that the next release (ORT 1.10) will require explicitly setting the providers parameter if you want to use execution providers other than the default CPU provider (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.*

### C/C++ API
All the options shown below are passed to SessionOptionsAppendExecutionProvider_OpenVINO() API and populated in the struct OrtOpenVINOProviderOptions in an example shown below, for example for CPU device type:

```
OrtOpenVINOProviderOptions options;
options.device_type = "CPU_FP32";
options.enable_vpu_fast_compile = 0;
options.device_id = "";
options.num_of_threads = 8;
options.use_compiled_network = false;
options.blob_dump_path = "";
options.context = 0x123456ff;
SessionOptionsAppendExecutionProvider_OpenVINO(session_options, &options);
```

### Summary of options
The following table lists all the available configuration options and the Key-Value pairs to set them:

| **Key** | **Key type** | **Allowable Values** | **Value type** | **Description** |
| --- | --- | --- | --- | --- |
| device_type | string | CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32, Any valid Hetero combination, Any valid Multi or Auto devices combination | string | Overrides the accelerator hardware type and precision with these values at runtime. If this option is not explicitly set, default hardware and precision specified during build time is used. |Overrides the accelerator hardware type and precision with these values at runtime. If this option is not explicitly set, default hardware and precision specified during build time is used. |
| device_id   | string | Any valid OpenVINO device ID | string | Selects a particular hardware device for inference. The list of valid OpenVINO device ID's available on a platform can be obtained either by Python API (`onnxruntime.capi._pybind_state.get_available_openvino_device_ids()`) or by [OpenVINO C/C++ API](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html#acb212aa879e1234f51b845d2befae41c). If this option is not explicitly set, an arbitrary free device will be automatically selected by OpenVINO runtime.|
| enable_vpu_fast_compile | string | True/False | boolean | This option is only available for MYRIAD_FP16 VPU devices. During initialization of the VPU device with compiled model, Fast-compile may be optionally enabled to speeds up the model's compilation to VPU device specific format. This in-turn speeds up model initialization time. However, enabling this option may slowdown inference due to some of the optimizations not being fully applied, so caution is to be exercised while enabling this option. |
| num_of_threads | string | Any unsigned positive number other than 0 | size_t | Overrides the accelerator default value of number of threads with this value at runtime. If this option is not explicitly set, default value of 8 is used during build time. |
| use_compiled_network | string | True/False | boolean | This option is only available for MYRIAD_FP16 VPU devices for both Linux and Windows and it enables save/load blob functionality. It can be used to directly import pre-compiled blobs if exists or dump a pre-compiled blob at the executable path. |
| blob_dump_path | string | Any valid string path on the hardware target | string | Explicitly specify the path where you would like to dump and load the blobs for the save/load blob feature when use_compiled_network setting is enabled . This overrides the default path.|
| context | string | OpenCL Context | void* | This option is only alvailable when OpenVINO EP is built with OpenCL flags enabled. It takes in the remote context i.e the cl_context address as a void pointer.|

Valid Hetero or Multi or Auto Device combinations:
HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...
The <DEVICE_TYPE> can be any of these devices from this list ['CPU','GPU','MYRIAD','FPGA','HDDL']

A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO or Multi-Device Build.

Example:
HETERO:MYRIAD,CPU  AUTO:GPU,CPU  MULTI:MYRIAD,GPU,CPU

### Other configuration settings
#### Onnxruntime Graph Optimization level
OpenVINO backend performs both hardware dependent as well as independent optimizations to the graph to infer it with on the target hardware with best possible performance. In most of the cases it has been observed that passing in the graph from the input model as is would lead to best possible optimizations by OpenVINO. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime before handing the graph over to OpenVINO backend. This can be done using Session options as shown below:-

#### Python API
```
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = onnxruntime.InferenceSession(<path_to_model_file>, options)
```

#### C/C++ API
```
SessionOptions::SetGraphOptimizationLevel(ORT_DISABLE_ALL);
```

#### Deprecated: Dynamic device type selection
**Note: This API has been deprecated. Please use the mechanism mentioned above to set the 'device-type' option.**
When ONNX Runtime is built with OpenVINO Execution Provider, a target hardware option needs to be provided. This build time option becomes the default target harware the EP schedules inference on. However, this target may be overriden at runtime to schedule inference on a different hardware as shown below.

Note. This dynamic hardware selection is optional. The EP falls back to the build-time default selection if no dynamic hardware option value is specified.

**Python API**
```
import onnxruntime
onnxruntime.capi._pybind_state.set_openvino_device("<harware_option>")
# Create session after this
```
*This property persists and gets applied to new sessions until it is explicity unset. To unset, assign a null string ("").*

**C/C++ API**

Append the settings string "<hardware_option>" to the EP settings string. Example shown below for the CPU_FP32 option:

```
std::string settings_str;
...
settings_str.append("CPU_FP32");
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(sf, settings_str.c_str()));
```

## Support Coverage

**ONNX Layers supported using OpenVINO**

The table below shows the ONNX layers supported and validated using OpenVINO Execution Provider.The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. VPU refers to USB based Intel<sup>®</sup> Movidius<sup>TM</sup>
VPUs as well as Intel<sup>®</sup> Vision accelerator Design with Intel Movidius <sup>TM</sup> MyriadX VPU.

| **ONNX Layers** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- |
| Abs | Yes | Yes | No |
| Acos | Yes | No | No |
| Acosh | Yes | No | No |
| Add | Yes | Yes | Yes |
| ArgMax | Yes | No | No |
| ArgMin | Yes | No | No |
| Asin | Yes | Yes | No |
| Asinh | Yes | Yes | No |
| Atan | Yes | Yes | No |
| Atanh | Yes | No | No |
| AveragePool | Yes | Yes | Yes |
| BatchNormalization | Yes | Yes | Yes |
| Ceil | Yes | Yes | Yes |
| Cast | Yes | Yes | Yes |
| Clip | Yes | Yes | Yes |
| Concat | Yes | Yes | Yes |
| Constant | Yes | Yes | Yes |
| ConstantOfShape | Yes | Yes | Yes |
| Conv | Yes | Yes | Yes |
| ConvTranspose | Yes | Yes | Yes |
| Cos | Yes | No | No |
| Cosh | Yes | No | No |
| DepthToSpace | Yes | Yes | Yes |
| DequantizeLinear | Yes | Yes | No |
| Div | Yes | Yes | Yes |
| Dropout | Yes | Yes | Yes |
| Elu | Yes | Yes | Yes |
| Equal | Yes | Yes | Yes |
| Erf | Yes | Yes | Yes |
| Exp | Yes | Yes | Yes |
| Expand | No | No | Yes |
| Flatten | Yes | Yes | Yes |
| Floor | Yes | Yes | Yes |
| Gather | Yes | Yes | Yes |
| GatherElements | No | No | Yes |
| GatherND | Yes | Yes | Yes |
| Gemm | Yes | Yes | Yes |
| GlobalAveragePool | Yes | Yes | Yes |
| GlobalLpPool | Yes | Yes | No |
| HardSigmoid | Yes | Yes | No |
| Identity | Yes | Yes | Yes |
| InstanceNormalization | Yes | Yes | Yes |
| LeakyRelu | Yes | Yes | Yes |
| Less | Yes | Yes | Yes |
| Log | Yes | Yes | Yes |
| Loop | Yes | Yes | Yes |
| LRN | Yes | Yes | Yes |
| MatMul | Yes | Yes | Yes |
| Max | Yes | Yes | Yes |
| MaxPool | Yes | Yes | Yes |
| Mean | Yes | Yes | Yes |
| Min | Yes | Yes | Yes |
| Mul | Yes | Yes | Yes |
| Neg | Yes | Yes | Yes |
| NonMaxSuppression | No | No | Yes |
| NonZero | Yes | No | Yes |
| Not | Yes | Yes | Yes |
| OneHot | Yes | Yes | Yes |
| Pad | Yes | Yes | Yes |
| Pow | Yes | Yes | Yes |
| PRelu | Yes | Yes | Yes |
| QuantizeLinear | Yes | Yes | No |
| Reciprocal | Yes | Yes | Yes |
| ReduceLogSum | Yes | No | Yes |
| ReduceMax | Yes | Yes | Yes |
| ReduceMean | Yes | Yes | Yes |
| ReduceMin | Yes | Yes | Yes |
| ReduceProd | Yes | No | No |
| ReduceSum | Yes | Yes | Yes |
| ReduceSumSquare | Yes | No | Yes |
| Relu | Yes | Yes | Yes |
| Reshape | Yes | Yes | Yes |
| Resize | Yes | No | Yes |
| RoiAlign | No | No | Yes |
| Round | Yes | Yes | Yes |
| Scatter | No | No | Yes |
| Selu | Yes | Yes | No |
| Shape | Yes | Yes | Yes |
| Sigmoid | Yes | Yes | Yes |
| Sign | Yes | No | No |
| SinFloat | No | No | Yes |
| Sinh | Yes | No | No |
| Slice | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes |
| Softsign | Yes | No | No |
| SpaceToDepth | Yes | Yes | Yes |
| Split | Yes | Yes | Yes |
| Sqrt | Yes | Yes | Yes |
| Squeeze | Yes | Yes | Yes |
| Sub | Yes | Yes | Yes |
| Sum | Yes | Yes | Yes |
| Tan | Yes | Yes | No |
| Tanh | Yes | Yes | Yes |
| Tile | Yes | Yes | Yes |
| TopK | Yes | Yes | Yes |
| Transpose | Yes | Yes | Yes |
| Unsqueeze | Yes | Yes | Yes |

### Topology Support

Below topologies from ONNX open model zoo are fully supported on OpenVINO Execution Provider and many more are supported through sub-graph partitioning

### Image Classification Networks

| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| bvlc_alexnet | Yes | Yes | Yes | Yes* |
| bvlc_googlenet | Yes | Yes | Yes | Yes* |
| bvlc_reference_caffenet | Yes | Yes | Yes | Yes* |
| bvlc_reference_rcnn_ilsvrc13 | Yes | Yes | Yes | Yes* |
| emotion ferplus | Yes | Yes | Yes | Yes* |
| densenet121 | Yes | Yes | Yes | Yes* |
| inception_v1 | Yes | Yes | Yes | Yes* |
| inception_v2 | Yes | Yes | Yes | Yes* |
| mobilenetv2 | Yes | Yes | Yes | Yes* |
| resnet18v1 | Yes | Yes | Yes | Yes* |
| resnet34v1 | Yes | Yes | Yes | Yes* |
| resnet101v1 | Yes | Yes | Yes | Yes* |
| resnet152v1 | Yes | Yes | Yes | Yes* |
| resnet18v2 | Yes | Yes | Yes | Yes* |
| resnet34v2 | Yes | Yes | Yes | Yes* |
| resnet101v2 | Yes | Yes | Yes | Yes* |
| resnet152v2 | Yes | Yes | Yes | Yes* |
| resnet50 | Yes | Yes | Yes | Yes* |
| resnet50v2 | Yes | Yes | Yes | Yes* |
| shufflenet | Yes | Yes | Yes | Yes* |
| squeezenet1.1 | Yes | Yes | Yes | Yes* |
| vgg19 | Yes | Yes | Yes | Yes* |
| vgg16 | Yes | Yes | Yes | Yes* |
| zfnet512 | Yes | Yes | Yes | Yes* |
| arcface | Yes | Yes | Yes | Yes* |


### Image Recognition Networks

| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| mnist | Yes | Yes | Yes | Yes* |

### Object Detection Networks

| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| tiny_yolov2 | Yes | Yes | Yes | Yes* |
| yolov3 | Yes | Yes | Yes | No* |
| tiny_yolov3 | Yes | Yes | Yes | No* |
| mask_rcnn | Yes | Yes | Yes | No* |
| faster_rcnn | Yes | Yes | Yes | No* |
| yolov4 | Yes | Yes | Yes | No* |
| yolov5 | Yes | Yes | Yes | No* |

### Image Manipulation Networks

| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| mosaic | Yes | Yes | Yes | No* |
| candy | Yes | Yes | Yes | No* |
| cgan | Yes | Yes | Yes | No* |
| rain_princess | Yes | yes | Yes | No* |
| pointilism | Yes | Yes | Yes | No* |
| udnie | Yes | Yes | Yes | No* |

*FPGA only runs in HETERO mode wherein the layers that are not supported on FPGA fall back to OpenVINO CPU.

## OpenVINO-EP samples Tutorials

In order to showcase what you can do with the OpenVINO Execution Provider for ONNX Runtime, we have created a few samples that shows how you can get that performance boost you’re looking for with just one additional line of code. 

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

