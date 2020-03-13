# OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#openvino).

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

The table below shows the ONNX layers supported using OpenVINO Execution Provider and the mapping between ONNX layers and OpenVINO layers. The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
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
| CropAndResize | Yes | Yes | Yes |
| Div | Yes | Yes | Yes |
| Dropout | Yes | Yes | Yes |
| Expand | Yes | Yes | Yes |
| Flatten | Yes | Yes | Yes |
| Floor | Yes | Yes | Yes |
| Gather | Yes | Yes | Yes |
| GatherND | Yes | Yes | Yes |
| Gemm | Yes | Yes | Yes |
| GlobalAveragePool | Yes | Yes | Yes |
| Identity | Yes | Yes | Yes |
| ImageScaler | Yes | Yes | Yes |
| LeakyRelu | Yes | Yes | Yes |
| Log | Yes | Yes | Yes |
| Loop | Yes | Yes | Yes |
| LRN | Yes | Yes | Yes |
| LSTM | Yes | Yes | Yes |
| MatMul | Yes | Yes | Yes |
| Max | Yes | Yes | Yes |
| MaxPool | Yes | Yes | Yes |
| Min | Yes | Yes | Yes |
| Mul | Yes | Yes | Yes |
| NonMaxSuppression | Yes | Yes | Yes |
| NonZero | Yes | Yes | Yes |
| OneHot | Yes | Yes | Yes |
| Pad | Yes | Yes | Yes |
| Pow | Yes | Yes | Yes |
| PRelu | Yes | Yes | Yes |
| Range | Yes | Yes | Yes |
| ReduceMax | Yes | Yes | Yes |
| ReduceMean | Yes | Yes | Yes |
| ReduceMin | Yes | Yes | Yes |
| ReduceSum | Yes | Yes | Yes |
| Relu | Yes | Yes | Yes |
| Reshape | Yes | Yes | Yes |
| Resize | Yes | Yes | Yes |
| RoiAlign | Yes | Yes | Yes |
| Scatter | Yes | Yes | Yes |
| Shape | Yes | Yes | Yes |
| Sigmoid | Yes | Yes | Yes |
| Slice | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes |
| Squeeze | Yes | Yes | Yes |
| Sub | Yes | Yes | Yes |
| Sum | Yes | Yes | Yes |
| Tanh | Yes | Yes | Yes |
| Tile | Yes | Yes | Yes |
| TopK | Yes | Yes | Yes |
| Transpose | Yes | Yes | Yes |
| Unsqueeze | Yes | Yes | Yes |
| Upsample | Yes | Yes | Yes |


## Topology Support

Below topologies are supported from ONNX open model zoo using OpenVINO Execution Provider

| **OPSET** | **MODEL NAME** | **CPU_FP32_UEP** | **GPU_FP16_UEP** | **GPU_FP32_UEP** | **MYRIAD_FP16_UEP** |
| --- | --- | --- | --- | --- | --- |
| opset10 | tf_inception_v4 | Yes | Yes | Yes | Yes |
| opset11 | tf_inception_v4 | Yes | Yes | Yes | Yes |
| opset7 | test_bvlc_alexnet | Yes | Yes | Yes | Yes |
| opset7 | test_bvlc_googlenet | Yes | Yes | Yes | Yes |
| opset7 | test_bvlc_reference_caffenet | Yes | Yes | Yes | Yes |
| opset7 | test_bvlc_reference_rcnn_ilsvrc13 | Yes | Yes | Yes | Yes |
| opset7 | test_densenet121 | Yes | Yes | Yes | Yes |
| opset7 | test_inception_v1 | Yes | Yes | Yes | Yes |
| opset7 | test_inception_v2 | Yes | Yes | Yes | Yes |
| opset7 | test_mnist | Yes | Yes | Yes | Yes |
| opset7 | test_mobilenetv2-1.0 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet101v2 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet152v2 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet18v2 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet34v2 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet50 | Yes | Yes | Yes | Yes |
| opset7 | test_resnet50v2 | Yes | Yes | Yes | Yes |
| opset7 | test_shufflenet | Yes | Yes | Yes | Yes |
| opset7 | test_squeezenet1.1 | Yes | Yes | Yes | Yes |
| opset7 | test_vgg19 | Yes | Yes | Yes | Yes |
| opset7 | test_zfnet512 | Yes | Yes | Yes | Yes |
| opset7 | tf_nasnet_large | Yes | Yes | Yes | Yes |
| opset7 | tf_nasnet_mobile | Yes | Yes | Yes | Yes |
| opset7 | tf_pnasnet_large | Yes | Yes | Yes | Yes |
| opset8 | mxnet_arcface | Yes | Yes | Yes | Yes |
| opset8 | test_bvlc_alexnet | Yes | Yes | Yes | Yes |
| opset8 | test_bvlc_googlenet | Yes | Yes | Yes | Yes |
| opset8 | test_bvlc_reference_caffenet | Yes | Yes | Yes | Yes |
| opset8 | test_bvlc_reference_rcnn_ilsvrc13 | Yes | Yes | Yes | Yes |
| opset8 | test_densenet121 | Yes | Yes | Yes | Yes |
| opset8 | test_inception_v1 | Yes | Yes | Yes | Yes |
| opset8 | test_inception_v2 | Yes | Yes | Yes | Yes |
| opset8 | test_mnist | Yes | Yes | Yes | Yes |
| opset8 | test_resnet50 | Yes | Yes | Yes | Yes |
| opset8 | test_shufflenet | Yes | Yes | Yes | Yes |
| opset8 | test_vgg19 | Yes | Yes | Yes | Yes |
| opset8 | test_zfnet512 | Yes | Yes | Yes | Yes |
| opset8 | tf_nasnet_large | Yes | Yes | Yes | Yes |
| opset8 | tf_nasnet_mobile | Yes | Yes | Yes | Yes |
| opset8 | tf_pnasnet_large | Yes | Yes | Yes | Yes |
| opset9 | cgan | Yes | Yes | Yes | No |
| opset9 | tf_inception_v4 | Yes | Yes | Yes | Yes |
| opset9 | tf_nasnet_large | Yes | Yes | Yes | Yes |
| opset9 | tf_nasnet_mobile | Yes | Yes | Yes | Yes |
| opset9 | tf_pnasnet_large | Yes | Yes | Yes | Yes |
