---
title: WinRT API
parent: API docs
grand_parent: Reference
---

# Windows Machine Learning WinRT API
{: .no_toc }

The ONNX Runtime Nuget package provides the ability to use the full [WinML API](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference).
This allows scenarios such as passing a [Windows.Media.VideoFrame](https://docs.microsoft.com/en-us/uwp/api/Windows.Media.VideoFrame) from your connected camera directly into the runtime for realtime inference.

The WinML API is a WinRT API that shipped inside the Windows OS starting with build 1809 (RS5) in the Windows.AI.MachineLearning namespace. It embedded a version of the ONNX Runtime.

In addition to using the in-box version of WinML, WinML can also be installed as an application redistributable package (see [layered architecture](../../resources/high-level-design.md#the-onnx-runtime-and-windows-os-integration) for technical details).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Supported Versions
Windows 8.1+

## Builds

|Artifact|Description|Supported Platforms|
|---|---|---|
|[Microsoft.AI.MachineLearning](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)|WinRT - CPU, GPU (DirectML)|Windows 8.1+|


## API Reference
[Windows.AI.MachineLearning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)

## Samples

Any code already written for the Windows.AI.MachineLearning API can be easily modified to run against the Microsoft.ML.OnnxRuntime package. All types originally referenced by inbox customers via the Windows namespace will need to be updated to now use the Microsoft namespace.

* [Samples in Github](https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/Desktop/cpp) 

## Should I use the in-box vs NuGet WinML version?

For a comparison, see [Windows Machine Learning: In-box vs NuGet WinML solutions](https://docs.microsoft.com/en-us/windows/ai/windows-ml/#in-box-vs-nuget-winml-solutions).

To detect if a particular OS version of Windows has the WinML APIs, use the [IsApiContractPresent](https://docs.microsoft.com/en-us/uwp/api/windows.foundation.metadata.apiinformation.isapicontractpresent) method.  This can be called from either UWP or native apps.

If the OS does not have the runtime you need you can switch to use the redist binaries instead.

|Release|API contract version|
|--|--|
|Windows OS 1809| 1|
|Windows OS 1903| 2|
|Windows OS 1909| 2|
|ORT release 1.2| 3|
|ORT release 1.3| 3|
|ORT release 1.4| 3|

See [here](https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions) for more about opsets and ONNX version details in Windows OS distributions.
