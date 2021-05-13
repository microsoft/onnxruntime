---
title: Compatibility
parent: Resources
toc: true
nav_order: 1
---

# ONNX Runtime compatibility
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Backwards compatibility
Newer versions of ONNX Runtime support all models that worked with prior versions, so updates should not break integrations.

## Environment compatibility
ONNX Runtime is not explicitly tested with every variation/combination of environments and dependencies, so this list is not comprehensive. Please use this as starting reference. For specific questions or requests, please [file an issue](https://github.com/microsoft/onnxruntime/issues) on Github.


### Platforms

* Windows

  * Tested with Windows 10 and Windows Server 2019
  * May be compatible with Windows 7+
  * Windows Machine Learning ([WinRT](https://www.onnxruntime.ai/docs/reference/api/winrt-api.html))
    * CPU: Windows 8.1+
    * GPU: Windows 10 1709+

* Linux
  * Tested with CentOS 7
  * Should be compatible with [distributions supported by .NET Core](https://docs.microsoft.com/en-us/dotnet/core/install/linux)

* Mac
  * Tested with 10.14 (Mojave)
  * May be compatible with 10.12+ (Sierra)

* Android
  * Tested with API level 28 (v9 "Pie")
  * May be compatible with API level 21+ (v5 "Lollipop")

* iOS
  * Tested with iOS 12
  * May be compatible with any 64bit iOS version (5S+)

### Compilers
* Windows 10: Visual C++ 2019
* Linux: gcc>=4.8

### Dependent Libraries
* [Submodules](https://github.com/microsoft/onnxruntime/tree/master/cgmanifests)
* See the [Execution Provider page](https://www.onnxruntime.ai/docs/reference/execution-providers/) for details on specific hardware libary version requirements


## ONNX opset support
ONNX Runtime supports all opsets from the latest released version of the [ONNX](https://onnx.ai) spec. All versions of ONNX Runtime support ONNX opsets from ONNX v1.2.1+ (opset version 7 and higher). 
  * For example: if an ONNX Runtime release implements ONNX opset 9, it can run models stamped with ONNX opset versions in the range [7-9]. 



* [Supported Operator Data Types](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md)
  * *Operators not supported in the current ONNX spec may be available as a [Contrib Operator](https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md)*
  * [How to add a custom operator/kernel](../how-to/add-custom-op.md)

| ONNX Runtime version | [ONNX version](https://github.com/onnx/onnx/blob/master/docs/Versioning.md) | ONNX opset version | ONNX ML opset version | ONNX IR version | [Windows ML Availability](https://docs.microsoft.com/en-us/windows/ai/windows-ml/release-notes/)|
|------------------------------|--------------------|--------------------|----------------------|------------------|------------------|
| 1.7.2<br/>1.7.1<br/>1.7.0 | **1.8** | 13 | 2 | 7 | Windows AI 1.7+ |
| 1.6.0 | **1.8** | 13 | 2 | 7 | Windows AI 1.6+ |
| 1.5.3<br/>1.5.2<br/>1.5.1 | **1.7** | 12 | 2 | 7 | Windows AI 1.5+ |
| 1.4.0 | **1.7** | 12 | 2 | 7 | Windows AI 1.4+ |
| 1.3.1<br/>1.3.0 | **1.7** | 12 | 2 | 7 | Windows AI 1.3+ |
| 1.2.0<br/>1.1.2<br/>1.1.1<br/>1.1.0 | **1.6** | 11 | 2 | 6 | Windows AI 1.3+ |
| 1.0.0 | **1.6** | 11 | 2 | 6 | Windows AI 1.3+ |
| 0.5.0 | **1.5** | 10 | 1 | 5 | Windows AI 1.3+ |
| 0.4.0 | **1.5** | 10 | 1 | 5 | Windows AI 1.3+ |
| 0.3.1<br/>0.3.0 | **1.4** | 9 | 1 | 3 | Windows 10 2004+ |
| 0.2.1<br/>0.2.0 | **1.3** | 8 | 1 | 3 | Windows 10 1903+ |
| 0.1.5<br/>0.1.4 | **1.3** | 8 | 1 | 3 | Windows 10 1809+ |

Unless otherwise noted, please use the latest released version of the tools to convert/export the ONNX model. Most tools are backwards compatible and support multiple ONNX versions. Join this with the table above to evaluate ONNX Runtime compatibility.


|Tool|Recommended Version|
|---|---|
|[PyTorch](https://pytorch.org/)|[Latest stable](https://pytorch.org/get-started/locally/)|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br>CoreML, LightGBM, XGBoost, LibSVM|[Latest stable](https://github.com/onnx/onnxmltools/releases)|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br> SparkML|[Latest stable](https://github.com/onnx/onnxmltools/releases)|
|[SKLearn-ONNX](https://pypi.org/project/skl2onnx/)|[Latest stable](https://github.com/onnx/sklearn-onnx/releases)|
|[Keras-ONNX](https://pypi.org/project/keras2onnx/)|[Latest stable](https://github.com/onnx/keras-onnx/releases)|
|[Tensorflow-ONNX](https://pypi.org/project/tf2onnx/)|[Latest stable](https://github.com/onnx/tensorflow-onnx/releases)|
|[WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools)|[Latest stable](https://pypi.org/project/winmltools/)|
|[AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)|[1.0.39+](https://pypi.org/project/azureml-automl-core) (ONNX v1.5) <br/>[1.0.33](https://pypi.org/project/azureml-automl-core/1.0.33/) (ONNX v1.4) |

