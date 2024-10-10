---
title: Compatibility
parent: Reference
toc: true
nav_order: 2
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
ONNX Runtime is not explicitly tested with every variation/combination of environments and dependencies, so this list is not comprehensive. Please use this as starting reference. For specific questions or requests, please [file an issue](https://github.com/microsoft/onnxruntime/issues) on GitHub.


### Platforms

* Windows
  * Tested with Windows Server 2022 and Windows 11.
  * May be compatible with Windows 10 and above.

* Linux
  * Tested with UBI8 (which is equivalent to Red Hat Enterprise Linux 8)
  * The running environment should have Glibc library with version >= 2.28. For example, Debian 10 and newer, Red Hat Enterprise Linux 8 and newer, Ubuntu 18.10 and newer, AzureLinux(Mariner Linux 2), Fedora 29 and newer. 

* Mac
  * Tested with 13 and 14
  * May be compatible with 11+

* Android
  * Tested with API level 28 (v9 "Pie")
  * May be compatible with API level 21+ (v5 "Lollipop")

* iOS
  * Tested with iOS 12
  * May be compatible with any 64bit iOS version (5S+)

### Compilers
* Windows 10: Visual C++ 2022
* Linux: gcc>=9

### Dependent Libraries
* [Submodules](https://github.com/microsoft/onnxruntime/tree/main/cgmanifests)
* See the [Execution Provider page](../execution-providers) for details on specific hardware libary version requirements


## ONNX opset support
ONNX Runtime supports all opsets from the latest released version of the [ONNX](https://onnx.ai) spec. All versions of ONNX Runtime support ONNX opsets from ONNX v1.2.1+ (opset version 7 and higher). 
  * For example: if an ONNX Runtime release implements ONNX opset 9, it can run models stamped with ONNX opset versions in the range [7-9]. 



* [Supported Operator Data Types](https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md)
  * *Operators not supported in the current ONNX spec may be available as a [Contrib Operator](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md)*
  * [How to add a custom operator/kernel](operators/add-custom-op.md)

| ONNX Runtime version | [ONNX version](https://github.com/onnx/onnx/blob/master/docs/Versioning.md) | ONNX opset version | ONNX ML opset version | ONNX IR version |
|------------------------------|--------------------|--------------------|----------------------|------------------|
| 1.18 | **1.16** | 21 | 4 | 10 |
| 1.17 | **1.15** | 20 | 4 | 9 |
| 1.16 | **1.14.1** | 19 | 3 | 9 |
| 1.15 | **1.14** | 19 | 3 | 8 |
| 1.14 | **1.13** | 18 | 3 | 8 |
| 1.13 | **1.12** | 17 | 3 | 8 |
| 1.12 | **1.12** | 17 | 3 | 8 |
| 1.11 | **1.11** | 16 | 2 | 8 |
| 1.10 | **1.10** | 15 | 2 | 8 |
| 1.9 | **1.10** | 15 | 2 | 8 |
| 1.8 | **1.9** | 14 | 2 | 7 |
| 1.7 | **1.8** | 13 | 2 | 7 |
| 1.6 | **1.8** | 13 | 2 | 7 |
| 1.5 | **1.7** | 12 | 2 | 7 |
| 1.4 | **1.7** | 12 | 2 | 7 |
| 1.3 | **1.7** | 12 | 2 | 7 |
| 1.2<br/>1.1 | **1.6** | 11 | 2 | 6 |
| 1.0 | **1.6** | 11 | 2 | 6 |
| 0.5 | **1.5** | 10 | 1 | 5 |
| 0.4 | **1.5** | 10 | 1 | 5 |
| 0.3 | **1.4** | 9 | 1 | 3 |
| 0.2 | **1.3** | 8 | 1 | 3 |
| 0.1 | **1.3** | 8 | 1 | 3 |

Unless otherwise noted, please use the latest released version of the tools to convert/export the ONNX model. Most tools are backwards compatible and support multiple ONNX versions. Join this with the table above to evaluate ONNX Runtime compatibility.


|Tool|Recommended Version|
|---|---|
|[PyTorch](https://pytorch.org/)|[Latest stable](https://pytorch.org/get-started/locally/)|
|[Tensorflow-ONNX](https://pypi.org/project/tf2onnx/)|[Latest stable](https://github.com/onnx/tensorflow-onnx/releases)|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br>CatBoost, CoreML, LightGBM, XGBoost, LibSVM, SparkML|[Latest stable](https://github.com/onnx/onnxmltools/releases)|
|[SKLearn-ONNX](https://pypi.org/project/skl2onnx/)|[Latest stable](https://github.com/onnx/sklearn-onnx/releases)|
|[WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools)|[Latest stable](https://pypi.org/project/winmltools/)|
|[AzureML AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)|[1.0.39+](https://pypi.org/project/azureml-automl-core) (ONNX v1.5) <br/>[1.0.33](https://pypi.org/project/azureml-automl-core/1.0.33/) (ONNX v1.4) |
|[Paddle2ONNX](https://pypi.org/project/paddle2onnx/)| [Latest stable](https://github.com/PaddlePaddle/Paddle2ONNX/releases) |

