# Hardware Enabled with OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs and integrated GPUs. Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel CPUs and GPUs supported.

# ONNX Layers supported using OpenVINO

Below table shows the ONNX layers supported using OpenVINO Execution Provider and the mapping between ONNX layers and OpenVINO layers. The below table also lists the Intel hardware support for each of the layers.  CPU refers to Intel<sup>®</sup>
 Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics.

| **ONNX Layers** | **OpenVINO Layers** | **CPU** | **GPU** | **VPU** | 
| --- | --- | --- | --- | --- |
| Add | Eltwise (operation=sum) | Yes | Yes | Yes
| AveragePool | Pooling(pool\_method=avg) | Yes | Yes | Yes
| BatchNormalization | Scaleshift (can be fused into Convlution or Fully Connected) | Yes | Yes | Yes
| Concat  | Concat | Yes | Yes | Yes
| Conv | Convolution | Yes | Yes | Yes
| Dropout | Ignored | Yes | Yes | Yes
| Flatten  | Reshape | Yes | Yes | Yes
| Gemm | FullyConnected | Yes | Yes | Yes
| GlobalAveragePool | Pooling | Yes | Yes | Yes
| Identity | Ignored | Yes | Yes | Yes
| ImageScaler | ScaleShift  | Yes  | Yes  | Yes
| LRN  | Norm | Yes | Yes | Yes
| MatMul | FullyConnected | Yes | Yes | No
| MaxPool | Pooling(pool\_method=max) | Yes | Yes | Yes
| Mul | Eltwise(operation=mul) | Yes | Yes | Yes
| Relu |  ReLU  | Yes | Yes | Yes
| Reshape | Reshape | Yes | Yes | Yes
|  Softmax  | SoftMax | Yes | Yes | Yes
| Sum | Eltwise(operation=sum) | Yes | Yes | Yes
| Transpose | Permute | Yes | Yes | Yes
| UnSqueeze | Reshape  | Yes  | Yes  | Yes
| LeakyRelu | ReLU | Yes  | Yes  | Yes


# Topology Support

Below topologies are supported from ONNX open model zoo using OpenVINO Execution Provider

## Image Classification Networks

| **Topology** | **CPU** | **GPU** | **VPU** | 
| --- | --- | --- | --- |  
| bvlc\_alexnet | Yes | Yes | Yes
| bvlc\_googlenet | Yes | Yes | Yes
| bvlc\_reference\_caffenet | Yes | Yes | Yes 
| bvlc\_reference\_rcnn\_ilsvrc13 | Yes | Yes | Yes 
| densenet121 | Yes | Yes | Yes
| Inception\_v1 | Yes | Yes | No
| Inception\_v2 | Yes | Yes | Yes
| Shufflenet | Yes | Yes | Yes
| Zfnet512 | Yes | Yes | Yes 
| Squeeznet 1.1 | Yes | Yes | Yes
| Resnet18v1 | Yes | Yes | Yes
| Resnet34v1 | Yes | Yes | Yes
| Resnet50v1 | Yes | Yes | Yes
| Resnet101v1 | Yes | Yes | Yes
| Resnet152v1 | Yes | Yes | Yes
| Resnet18v2  | Yes | Yes | Yes
| Resnet34v2  | Yes | Yes | Yes
| Resnet50v2  | Yes | Yes | Yes
| Resnet101v2 | Yes | Yes | Yes
| Resnet152v2 | Yes | Yes | Yes 
| Mobilenetv2 | Yes | Yes | Yes
| vgg16       | Yes | Yes | Yes
| vgg19       | Yes | Yes | Yes

## Image Recognition Networks

| **Topology** | **CPU** | **GPU** | **VPU** | 
| --- | --- | --- | --- | 
| MNIST | Yes | Yes | No

## Object Detection Networks

| **Topology** | **CPU** | **GPU** | **VPU** | 
| --- | --- | --- | --- | 
|TinyYOLOv2* | Yes | Yes | Yes
| ResNet101\_DUC\_HDC | Yes | Yes | No 

*Not supported by default on OpenVINO Execution Provider, please follow the instructions from the section on Dynamic Input Shapes to enable the support for TinyYOLOv2.

# Support for Dynamic Input Shapes

Some deep learning models converted to ONNX have dynamic input shapes. Dynamic shapes are represented with 'None' in the first dimension of the input. TinyYOLOv2 is an example model with dynamic input shapes. Networks with dynamic shapes are not supported by default on OpenVINO Execution Provider. The execution falls back to the default CPU Execution Provider. The support for networks with dynamic input shapes will be coming soon in the OpenVINO Execution Provider. Meanwhile, the users can enable networks with dynamic shapes using a custom build to leverage optimiations from OpenVINO. Below are the instructions to build the execution provider for enabling models with dynamic input shapes.

Edit the below variables in the file 'openvino_execution_provider.h' in $onnxruntimeroot/onnxruntime/core/providers/openvino

- Set the variable <code>ENABLE_DYNAMIC_INPUT_SHAPE</code> to 1

- Set the variable <code>DYNAMIC_DIMENSION</code> to the first input dimension of the dynamic shape. The value 'None' in the first dimension of the input will be replaced by the first dimension value provided by the user.

- Follow the instructions in BUILD.md for rebuilding the OpenVINO execution provider

# Application code changes for VAD-R performance scaling

Batch the input images at the application code level for scaling the performance on HDDL-R with 8 VPUs:

Sample Code Snippet:

<code> for img in images: </code>

<code>  img = cv2.resize(img, (224,224)) </code>

<code>  x = numpy.asarray(img).astype(numpy.float32) </code>

<code>  x = numpy.transpose(x, (2,0,1)) </code>

<code>  orig_shape = x.shape </code>

<code>  x = numpy.expand_dims(x,axis=0) </code>

<code>  if y is None: </code>

<code>     y = x </code>

<code>  else: </code>

<code>     y = numpy.concatenate((y,x), axis=0) </code>

Output results will be batched as well. Post-processing steps need to be added depending on the type of topology used (classification/object detection/etc.)

