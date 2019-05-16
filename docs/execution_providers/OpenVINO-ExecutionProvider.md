# Hardware Enabled with OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs and integrated GPUs. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel CPUs and GPUs supported.

# ONNX Layers supported using OpenVINO

Below table shows the ONNX layers supported using OpenVINO Execution Provider and the mapping between ONNX layers and OpenVINO layers. The below table also lists the Intel hardware support for each of the layers.  CPU refers to Intel<sup>®</sup>
 Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics.

| **ONNX Layers** | **OpenVINO Layers** | **CPU** | **GPU** |
| --- | --- | --- | --- |
| Add | Eltwise (operation=sum) | Yes | Yes |
| AveragePool | Pooling(pool\_method=avg) | Yes | Yes |
| BatchNormalization | Scaleshift (can be fused into Convlution or Fully Connected) | Yes | Yes |
| Concat  | Concat | Yes | Yes |
| Conv | Convolution | Yes | Yes |
| Dropout | Ignored | Yes | Yes |
| Flatten  | Reshape | Yes | Yes |
| Gemm | FullyConnected | Yes | Yes |
| GlobalAveragePool | Pooling | Yes | Yes |
| Identity | Ignored | Yes | Yes |
| ImageScaler | ScaleShift  | Yes  | Yes  |
| LRN  | Norm | Yes | Yes |
| MatMul | FullyConnected | Yes | Yes |
| MaxPool | Pooling(pool\_method=max) | Yes | Yes |
| Mul | Eltwise(operation=mul) | Yes | Yes |
| Relu |  ReLU  | Yes | Yes |
| Reshape | Reshape | Yes | Yes |
|  Softmax  | SoftMax | Yes | Yes |
| Sum | Eltwise(operation=sum) | Yes | Yes |
| Transpose | Permute | Yes | Yes |
| UnSqueeze | Reshape  | Yes  | Yes  |
| LeakyRelu | ReLU | Yes  | Yes  |


# Topology Support

Below topologies are supported from ONNX open model zoo using OpenVINO Execution Provider

## Image Classification Networks

- bvlc\_alexnet

- bvlc\_googlenet

- bvlc\_reference\_caffenet

- bvlc\_reference\_rcnn\_ilsvrc13

- densenet121

- Inception\_v1

- Inception\_v2

- Shufflenet

- Zfnet512

- Squeeznet 1.1

- Resnet18v1

- Resnet34v1

- Resnet50v1

- Resnet101v1

- Resnet152v1

- Resnet18v2

- Resnet34v2

- Resnet50v2

- Resnet101v2

- Resnet152v2

- Mobilenetv2

- vgg16

- vgg19


## Image Recognition Networks

- MNIST

## Object Detection Networks

- TinyYOLOv2 (Not supported by default on OpenVINO Execution Provider, please follow the instructions from the section on Dynamic Input Shapes to enable the support for TinyYOLOv2.)

- ResNet101\_DUC\_HDC

# Support for Dynamic Input Shapes

Some deep learning models converted to ONNX have dynamic input shapes. Dynamic shapes are represented with 'None' in the first dimension of the input. TinyYOLOv2 is an example model with dynamic input shapes. Networks with dynamic shapes are not supported by default on OpenVINO Execution Provider. The execution falls back to the default CPU Execution Provider. The support for networks with dynamic input shapes will be coming soon in the OpenVINO Execution Provider. Meanwhile, the users can enable networks with dynamic shapes using a custom build to leverage optimiations from OpenVINO. Below are the instructions to build the execution provider for enabling models with dynamic input shapes.

Edit the below variables in the file 'openvino_execution_provider.h' in $onnxruntimeroot/onnxruntime/core/providers/openvino

- Set the variable <code>ENABLE_DYNAMIC_INPUT_SHAPE</code> to 1

- Set the variable <code>DYNAMIC_DIMENSION</code> to the first input dimension of the dynamic shape. The value 'None' in the first dimension of the input will be replaced by the first dimension value provided by the user.

- Follow the instructions in BUILD.md for rebuilding the OpenVINO execution provider

