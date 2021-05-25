This is a C# sample for demoing ONNXRuntime OpenVINO Execution Provider usage:

1. The object detection sample uses YOLOv3 Deep Learning ONNX Model from the ONNX Model Zoo.

2. The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNX RT to run inference on Intel<sup>®</sup> NCS2 stick (MYRIADX device). The sample uses ImageSharp for image processing and ONNX Runtime OpenVINO EP for inference.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime/tree/master/csharp/sample/OpenVINO_EP_samples/yolov3_object_detection).

# How to build

## Prerequisites
1. Install [.NET Core 3.1](https://dotnet.microsoft.com/download/dotnet-core/3.1) or higher for you OS (Mac, Windows or Linux).

2. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)

3. Use any sample Image as input to the sample.

4. Download the latest YOLOv3 model from the ONNX Model Zoo.

## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html#build)


## Reference Documentation
[Documentation](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html)

To build nuget packages of onnxruntime with openvino flavour
```
./build.sh --config Release --use_openvino MYRIAD_FP16 --build_shared_lib --build_nuget
```

## Build the sample C# Application

1. Create a new console project

```
dotnet new console
```

2. Install Nuget Packages of Onnxruntime and [ImageSharp](https://www.nuget.org/packages/SixLabors.ImageSharp)

    1. Right click on project, navigate to Manage Nuget Packages.
    2. Install Image Sharp Packages from nuget.org.
    3. Install Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Openvino from the build directory nuget-artifacts. 
    

3. Compile the sample

```
dotnet build
```

4. Run the sample

```
dotnet run [path-to-model] [path-to-image] [path-to-output-image]
```

## Link to Download the YOLOv3 ONNX Model:

This example was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [YOLOv3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) model from here.


## References:
[fasterrcnn_csharp](https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/tutorials/fasterrcnn_csharp.md)

[resnet50_csharp](https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/tutorials/resnet50_csharp.md)
