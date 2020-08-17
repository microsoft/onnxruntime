# OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#openvino).

## Onnxruntime Graph Optimization level
OpenVINO backend performs both hardware dependent as well as independent optimizations to the graph to infer it with on the target hardware with best possible performance. In most of the cases it has been observed that passing in the graph from the input model as is would lead to best possible optimizations by OpenVINO. For this reason, it is advised to turn off high level optimizations performed by ONNX Runtime before handing the graph over to OpenVINO backend. This can be done using Session options as shown below:-

1. Python API
```
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = onnxruntime.InferenceSession(<path_to_model_file>, options)
```

2. C++ API
```
SessionOptions::SetGraphOptimizationLevel(ORT_DISABLE_ALL);
```

## Dynamic device selection
When ONNX Runtime is built with OpenVINO Execution Provider, a target hardware option needs to be provided. This build time option becomes the default target harware the EP schedules inference on. However, this target may be overriden at runtime to schedule inference on a different hardware as shown below.

Note. This dynamic hardware selection is optional. The EP falls back to the build-time default selection if no dynamic hardware option value is specified.
1. Python API
```
import onnxruntime
onnxruntime.capi._pybind_state.set_openvino_device("<harware_option>")
# Create session after this
```
2. C/C++ API
```
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(sf, "<hardware_option>"));
```

## ONNX Layers supported using OpenVINO

The table below shows the ONNX layers supported and validated using OpenVINO Execution Provider.The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. VPU refers to USB based Intel<sup>®</sup> Movidius<sup>TM</sup>
VPUs as well as Intel<sup>®</sup> Vision accelerator Design with Intel Movidius <sup>TM</sup> MyriadX VPU.

| **ONNX Layers** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- |
| Add | Yes | Yes | Yes |
| ArgMax | Yes | Yes | Yes |
| AveragePool | Yes | Yes | Yes |
| BatchNormalization | Yes | Yes | Yes |
| Cast | Yes | Yes | Yes |
| Clip | Yes | Yes | Yes |
| Concat | Yes | Yes | Yes |
| Constant | Yes | Yes | Yes |
| Conv | Yes | Yes | Yes |
| ConvTranspose | Yes | Yes | Yes |
| Div | Yes | Yes | Yes |
| Dropout | Yes | Yes | Yes |
| Flatten | Yes | Yes | Yes |
| Floor | Yes | Yes | Yes |
| Gather | Yes | Yes | Yes |
| GatherND | Yes | Yes | Yes |
| Gemm | Yes | Yes | Yes |
| GlobalAveragePool | Yes | Yes | Yes |
| Identity | Yes | Yes | Yes |
| LeakyRelu | Yes | Yes | Yes |
| Log | Yes | Yes | Yes |
| LRN | Yes | Yes | Yes |
| LSTM | Yes | Yes | Yes |
| MatMul | Yes | Yes | Yes |
| Max | Yes | Yes | Yes |
| MaxPool | Yes | Yes | Yes |
| Min | Yes | Yes | Yes |
| Mul | Yes | Yes | Yes |
| Pad | Yes | Yes | Yes |
| Pow | Yes | Yes | Yes |
| PRelu | Yes | Yes | Yes |
| ReduceMax | Yes | Yes | Yes |
| ReduceMean | Yes | Yes | Yes |
| ReduceMin | Yes | Yes | Yes |
| ReduceSum | Yes | Yes | Yes |
| Relu | Yes | Yes | Yes |
| Reshape | Yes | Yes | Yes |
| Sigmoid | Yes | Yes | Yes |
| Slice | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes |
| Squeeze | Yes | Yes | Yes |
| Sub | Yes | Yes | Yes |
| Sum | Yes | Yes | Yes |
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

*FPGA only runs in HETERO mode wherein the layers that are not supported on FPGA fall back to OpenVINO CPU.