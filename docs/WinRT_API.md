# Windows Machine Learning WinRT API

New in the ONNX Runtime Nuget package is the ability to use the full [Windows.AI.MachineLearning API](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).

This allows scenarios such as passing a [Windows.Media.VideoFrame](https://docs.microsoft.com/en-us/uwp/api/Windows.Media.VideoFrame) from your connected camera directly into the runtime for realtime inference.

The Windows.AI.MachineLearning API is a WinRT API that shipped inside the Windows OS starting with build 1809 (RS5).   It embedded a version of the ONNX Runtime.

Many customers have asked for a way to use this offering as an application redistributable package.

With our new [layered architecture](HighLevelDesign.md#the-onnx-runtime-and-windows-os-integration) you can now do this, with some limitations.

## NuGet Package

The Microsoft.ML.OnnxRuntime [Nuget package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) includes the precompiled binaries for using the ONNX runtime with the WinRT API.   Support is compiled directly into *onnxruntime.dll*

Note: As of the 1.2 release, you can use all of the CPU functionality from these binaries.  In order to get GPU funtionality using DirectML, you will need to build the binary yourself using [these instructions](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#DirectML).

## Sample Code

Any code already written for the Windows.AI.MachineLearning API can be easily modified to run against the Microsoft.ML.OnnxRuntime package.  Check out these [existing samples](https://github.com/microsoft/windows-Machine-Learning) in github.

## Activation and Side-by-Side

Because Windows.AI.MachineLearning ships inside the OS, default object activation is going to use those OS binaries.  Applications must explicitly code to enable the use of the redist binaries when creating WinML objects (Like [LearningModelSession](https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning.learningmodelsession)).

Read up [here](HighLevelDesign.md#the-onnx-runtime-and-windows-os-integration) in how to decide when to use the OS binaries and when to use redist binaries.

To create objects using the redist binaries you have several choices depending on how you are consuming the WinRT:

* cpp/winrt:  You can use WINRT_RoGetActivationFactory hooking as shown [here](https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/SqueezeNetObjectDetection/Desktop/cpp/dllload.cpp) in our sample projects.
* WRL: (coming soon)
* Raw C++:  Simply use the similar code to the cpp/winrt sample to load and use the activation factory in your redist binary.

## Deciding which header files to use

The best way to use the API is to use the header files that come with the Windows SDK.  

* For Visual Studio they are included as an optional feature.
* For Visual Studio Code you can download them [here](https://developer.microsoft.com/en-US/windows/downloads/windows-10-sdk/).

This [tutorial](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop) is a great place to get started.

To detect if an OS already has Windows.AI.MachineLearning you can use the [IsApiContractPresent](https://docs.microsoft.com/en-us/uwp/api/windows.foundation.metadata.apiinformation.isapicontractpresent) method.  This can be called from either UWP or native apps.

If the OS does not have the runtime you need you can switch to use the redist binaries instead.

|Release|API contract version|
|--|--|
|Windows OS 1809| 1|
|Windows OS 1903| 2|
|Windows OS 1909| 2|
|ORT release 1.2| 3|

See [here](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions) for more about opsets and ONNX version details in Windows OS distributons.
