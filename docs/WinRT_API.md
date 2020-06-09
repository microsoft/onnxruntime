# Windows Machine Learning WinRT API

New in the ONNX Runtime Nuget package is the ability to use the full [WinML API](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).

This allows scenarios such as passing a [Windows.Media.VideoFrame](https://docs.microsoft.com/en-us/uwp/api/Windows.Media.VideoFrame) from your connected camera directly into the runtime for realtime inference.

The WinML API is a WinRT API that shipped inside the Windows OS starting with build 1809 (RS5) in the Windows.AI.MachineLearning namespace.   It embedded a version of the ONNX Runtime.

Many customers have asked for a way to use this offering as an application redistributable package.

With our new [layered architecture](InferenceHighLevelDesign.md#the-onnx-runtime-and-windows-os-integration) you can now do this, with some limitations. The WinML APIs have been lifted and mirrored into the Microsoft.AI.MachineLearning namespace in the redistributable.

## NuGet Package

The Microsoft.AI.MachineLearning [Nuget package](https://www.nuget.org/packages/Microsoft.AI.MachineLearning/) includes the precompiled binaries for using the ONNX runtime with the WinRT API.   Support is compiled directly into *onnxruntime.dll*

Note: As of the 1.3 release, you can use all of the CPU and GPU functionality from these binaries.

## Sample Code

Any code already written for the Windows.AI.MachineLearning API can be easily modified to run against the Microsoft.ML.OnnxRuntime package. All types originally referenced by inbox customers via the Windows namespace will need to be updated to now use the Microsoft namespace. Check out these [existing samples](https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/Desktop/cpp) in github.

## Deciding on whether to use WinML in the Windows SDK or the Redist
To detect if a particular OS version of Windows has the WinML APIs, use the [IsApiContractPresent](https://docs.microsoft.com/en-us/uwp/api/windows.foundation.metadata.apiinformation.isapicontractpresent) method.  This can be called from either UWP or native apps.

If the OS does not have the runtime you need you can switch to use the redist binaries instead.

|Release|API contract version|
|--|--|
|Windows OS 1809| 1|
|Windows OS 1903| 2|
|Windows OS 1909| 2|
|ORT release 1.2| 3|
|ORT release 1.3| 3|

See [here](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions) for more about opsets and ONNX version details in Windows OS distributions.
