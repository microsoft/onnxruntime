# Windows Machine Learning WinRT API
New in the ONNX Runtime Nuget package is the ability to use the full [Windows.AI.MachineLearning API](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).

This allows scenarios such as passing a [Windows.Media.VideoFrame](https://docs.microsoft.com/en-us/uwp/api/Windows.Media.VideoFrame) from your connected camera directly into the runtime for realtime inference.

The Windows.AI.MachineLearning API is a WinRT API that shipped inside the windows OS starting with build 1809 (RS5).   It embedded a version of the ONNX runtime.

Many customers have asked for a way to use this API (and embedded ONNX runtime) as an application redistributable package.

With our new [layered architecture]() you can now do this, with some limitations.

## NuGet Package
The Microsoft.ML.OnnxRuntime Nuget package includes the precompiled binaries for using the ONNX runtime with the WinRT API.   Support is compiled directly into *onnxruntime.dll*

Note: As of the 1.2 release, you can use all of the CPU functionality from these binaries.  In order to get GPU funtionality using DirectML, you will need to build the binary yourself using [these instructions](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#DirectML).

## Sample Code

Any code already written for the Windows.AI.MachineLearning API can be easily modified to run against the Microsoft.ML.OnnxRuntime package.  Check out these [existing samples](https://github.com/microsoft/windows-Machine-Learning) in github.

## Activation and Side-by-Side

Since the Windows.AI.MachineLearning ships inside the Windows OS as **system32\Windows.AI.MachineLearning.dll** , your application needs to take care in selecting which binary it wants to load and use.

Normal WinRT activation would use [RoActivateInstance](https://docs.microsoft.com/en-us/windows/win32/api/roapi/nf-roapi-roactivateinstance).  Applications must explicityly choose which binary they want to use when including a redist version.

Read up [here]() in how to decide which binary to us.

Once you have chosen redist versus system32, you must always use that binary.   Mix and match of system32 and redist binaries are not supported.

To activate inbox objects, continue to use RoActivateInstance.    To active redist objects, you have several choices depending on how you are consuming WinRT:
* cpp/winrt:  You can use WINRT_RoGetActivationFactory hooking to allow using the redist instead of system32.   Look [here](https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/SqueezeNetObjectDetection/Desktop/cpp/dllload.cpp) for a sample on how.
* WRL: (coming soon)
* Raw C++:  Simply use the similar code to the cpp/winrt sample to load and use the activation factory in your redist binary.

## Deciding which header files to use

The best way to use the API is to use the header files that come with the Windows SDK.  You can download Windows SDK's either using Visual Studio or by going [here](https://developer.microsoft.com/en-US/windows/downloads/windows-10-sdk/).

You use contract targeting just like you would with any WinRT API.

You need to take care to make sure you contract targets work with the OS or redist binaries that you choose to deploy with.    Using this combination you can know when the OS has the contract you need, or you can fallback to your redist package if needed.  Use the [IsApiContractPresent](https://docs.microsoft.com/en-us/uwp/api/windows.foundation.metadata.apiinformation.isapicontractpresent) method to do this.  This can be called from UWP and not UWP apps easily.