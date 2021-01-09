# OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

### Build
For build instructions, please see the [BUILD page](../../BUILD.md#openvino).

## Runtime configuration options
---

OpenVINO EP can be configured with certain options at runtime that control the behavior of the EP. These options can be set as key-value pairs as below:-

### Python API
Key-Value pairs for config options can be set using the Session.set_providers API as follows:-

```
session = onnxruntime.InferenceSession(<path_to_model_file>, options)
session.set_providers(['OpenVINOExecutionProvider'], [{Key1 : Value1, Key2 : Value2, ...}])
```
*Note that this causes the InferenceSession to be re-initialized, which may cause model recompilation and hardware re-initialization*

### C/C++ API
All the options shown below are passed to SessionOptionsAppendExecutionProvider_OpenVINO() API and populated in the struct OrtOpenVINOProviderOptions in an example shown below, for example for CPU device type:-

```
OrtOpenVINOProviderOptions options;
options.device_type = "CPU_FP32";
options.enable_vpu_fast_compile = 0;
options.device_id = "";
options.num_of_threads = 8;
SessionOptionsAppendExecutionProvider_OpenVINO(session_options, &options);
```

### Available configuration options
The following table lists all the available configuratoin optoins and the Key-Value pairs to set them:-

| **Key** | **Key type** | **Allowable Values** | **Value type** | **Description** |
| --- | --- | --- | --- | --- |
| device_type | string | CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32, Any valid Hetero combination, Any valid Multi-Device combination | string | Overrides the accelerator hardware type and precision with these values at runtime. If this option is not explicitly set, default hardware and precision specified during build time is used. |
| device_id   | string | Any valid OpenVINO device ID | string | Selects a particular hardware device for inference. The list of valid OpenVINO device ID's available on a platform can be obtained either by Python API (`onnxruntime.capi._pybind_state.get_available_openvino_device_ids()`) or by [OpenVINO C/C++ API](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html#acb212aa879e1234f51b845d2befae41c). If this option is not explicitly set, an arbitrary free device will be automatically selected by OpenVINO runtime.|
| enable_vpu_fast_compile | string | True/False | boolean | This option is only available for MYRIAD_FP16 VPU devices. During initialization of the VPU device with compiled model, Fast-compile may be optionally enabled to speeds up the model's compilation to VPU device specific format. This in-turn speeds up model initialization time. However, enabling this option may slowdown inference due to some of the optimizations not being fully applied, so caution is to be exercised while enabling this option. |
| num_of_threads | string | Any unsigned positive number other than 0 | size_t | Overrides the accelerator default value of number of threads with this value at runtime. If this option is not explicitly set, default value of 8 is used during build time. |

Valid Hetero or Multi-Device combination's:
HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...
MULTI:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...
The <DEVICE_TYPE> can be any of these devices from this list ['CPU','GPU','MYRIAD','FPGA','HDDL']

A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO or Multi-Device Build.

Example:
HETERO:MYRIAD,CPU  HETERO:HDDL,GPU,CPU  MULTI:MYRIAD,GPU,CPU

## Other configuration settings
### Onnxruntime Graph Optimization level
OpenVINO backend performs both hardware dependent as well as independent optimizations to the graph to infer it with on the target hardware with best possible performance. In most of the cases it has been observed that passing in the graph from the input model as is would lead to best possible optimizations by OpenVINO. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime before handing the graph over to OpenVINO backend. This can be done using Session options as shown below:-

### Python API
```
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = onnxruntime.InferenceSession(<path_to_model_file>, options)
```

### C/C++ API
```
SessionOptions::SetGraphOptimizationLevel(ORT_DISABLE_ALL);
```

### Deprecated: Dynamic device type selection
**Note: This API has been deprecated. Please use the mechanism mentioned above to set the 'device-type' option.**
When ONNX Runtime is built with OpenVINO Execution Provider, a target hardware option needs to be provided. This build time option becomes the default target harware the EP schedules inference on. However, this target may be overriden at runtime to schedule inference on a different hardware as shown below.

Note. This dynamic hardware selection is optional. The EP falls back to the build-time default selection if no dynamic hardware option value is specified.

### Python API
```
import onnxruntime
onnxruntime.capi._pybind_state.set_openvino_device("<harware_option>")
# Create session after this
```
*This property persists and gets applied to new sessions until it is explicity unset. To unset, assign a null string ("").*

### C/C++ API

Append the settings string "<hardware_option>" to the EP settings string. Example shown below for the CPU_FP32 option:
```
std::string settings_str;
...
settings_str.append("CPU_FP32");
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(sf, settings_str.c_str()));
```

## ONNX Layers supported using OpenVINO

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
| Gemm | Yes | Yes | Yes |
| GlobalAveragePool | Yes | Yes | Yes |
| GlobalLpPool | Yes | Yes | No |
| HardSigmoid | Yes | Yes | No |
| Identity | Yes | Yes | Yes |
| InstanceNormalization | Yes | Yes | Yes |
| LeakyRelu | Yes | Yes | Yes |
| Less | Yes | Yes | Yes |
| Log | Yes | Yes | Yes |
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
| Not | Yes | Yes | No |
| OneHot | Yes | Yes | Yes |
| Pad | Yes | Yes | Yes |
| Pow | Yes | Yes | Yes |
| PRelu | Yes | Yes | Yes |
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
| TopK | Yes | Yes | Yes |
| Transpose | Yes | Yes | Yes |
| Unsqueeze | Yes | Yes | Yes |

## Topology Support

Below topologies from ONNX open model zoo are fully supported on OpenVINO Execution Provider and many more are supported through sub-graph partitioning

## Image Classification Networks

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


## Image Recognition Networks
| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| mnist | Yes | Yes | Yes | Yes* |

## Object Detection Networks
| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| tiny_yolov2 | Yes | Yes | Yes | Yes* |

## Image Manipulation Networks
| **MODEL NAME** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- |
| mosaic | Yes | No | No | No* |
| candy | Yes | No | No | No* |
| rain_princess | Yes | No | No | No* |
| pointilism | Yes | No | No | No* |
| udnie | Yes | No | No | No* |

*FPGA only runs in HETERO mode wherein the layers that are not supported on FPGA fall back to OpenVINO CPU.

## CSharp API

To use csharp api for openvino execution provider create a custom nuget package. Follow the instructions [here](../../BUILD.md##build-nuget-packages) to install prerequisites for nuget creation. Once prerequisites are installed follow the instructions to [build openvino](../../BUILD.md#openvino) and add an extra flag `--build_nuget` to create nuget packages. Two nuget packages will be created Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Openvino.

## Multi-threading for OpenVINO EP

OpenVINO Execution Provider enables thread-safe deep learning inference

## Heterogeneous Execution for OpenVINO EP

The heterogeneous Execution enables computing for inference on one network on several devices. Purposes to execute networks in heterogeneous mode

To utilize accelerators power and calculate heaviest parts of network on accelerator and execute not supported layers on fallback devices like CPU
To utilize all available hardware more efficiently during one inference

For more information on Heterogeneous plugin of OpenVINO, please refer to the following
[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_HETERO.html).

## Multi-Device Execution for OpenVINO EP

Multi-Device plugin automatically assigns inference requests to available computational devices to execute the requests in parallel. Potential gains are as follows

Improved throughput that multiple devices can deliver (compared to single-device execution)
More consistent performance, since the devices can now share the inference burden (so that if one device is becoming too busy, another device can take more of the load)

For more information on Multi-Device plugin of OpenVINO, please refer to the following
[documentation](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_MULTI.html#introducing_multi_device_execution).
