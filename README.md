<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is an open-source scoring engine for Open Neural Network Exchange (ONNX) models.

ONNX is an open format for machine learning (ML) models that is supported by various ML and DNN frameworks and tools. This format makes it easier to interoperate between frameworks and to maximize the reach of your hardware optimization investments. Learn more about ONNX on [https://onnx.ai](https://onnx.ai) or view the [Github Repo](https://github.com/onnx/onnx).
***
**Setup**
* [Installation](#installation)
* [APIs and Official Binaries](#apis-and-official-builds)
* [Building from Source](#building-from-source)

**Getting Started**
* [Deploying ONNX Runtime](#deploying-onnx-runtime)
* [Getting ONNX Models](#getting-onnx-models)
* [Examples and Tutorials](#examples-and-tutorials)

**About ONNX Runtime**
* [Why use ONNX Runtime](#why-use-onnx-runtime)
* [Design and Key Features](#design-and-key-features)

**[Contributions and Feedback](#contribute)**

**[License](#license)**
***
## Installation
**Quick Start:** The [ONNX-Ecosystem Docker container image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem) is available on Dockerhub and includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started.

Additional dockerfiles for some features can be found [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles).

### System Requirements
* ONNX Runtime binaries in the CPU packages use OpenMP and depend on the library being available at runtime in the
system.
  * For Windows, OpenMP support comes as part of VC runtime. It is also available as redist packages:
    [vc_redist.x64.exe](https://aka.ms/vs/15/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/15/release/vc_redist.x86.exe)
  * For Linux, the system must have the libgomp.so.1 which can be installed using ```apt-get install libgomp1```.
* The official GPU builds require the CUDA 9.1 and cuDNN 7.1 runtime libraries being installed in the system.
* Python binaries are compatible with Python 3.5-3.7.
* Certain operators makes use of system locales. At the very least you will need to install English language package and configure en_US.UTF-8 locale.
  * For Ubuntu install language-pack-en package
  * Run the following commands:
  
    `locale-gen en_US.UTF-8`
    
    `update-locale LANG=en_US.UTF-8`
  * Follow similar procedure to configure other locales on other platforms.

## APIs and Official Builds

### APIs
* [Python](https://aka.ms/onnxruntime-python)
* [C](docs/C_API.md)
* [C#](docs/CSharp_API.md)
* [C++](onnxruntime/core/session/inference_session.h)

### Official Builds
| | CPU (MLAS+Eigen) | CPU (MKL-ML) | GPU (CUDA)
|---|:---|:---|:---|
| **Python** | **[pypi: onnxruntime](https://pypi.org/project/onnxruntime)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | -- | **[pypi: onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)**<br><br>Windows (x64)<br>Linux (x64) |
| **C#** | **[Nuget: Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/)**<br><br>Windows (x64, x86)<br>Linux (x64, x86)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.MKLML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)**<br><br>Windows (x64)<br>Linux (x64) |
| **C** | **[Nuget: Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime)**<br><br>**[.zip, .tgz](https://aka.ms/onnxruntime-release)**<br><br>Windows (x64, x86)<br>Linux (x64, x86)<br>Mac OS X (x64 | **[Nuget: Microsoft.ML.OnnxRuntime.MKLML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)**<br><br>**[.zip, .tgz](https://aka.ms/onnxruntime-release)**<br><br>Windows (x64)<br>Linux (x64) |

## Building from Source
If additional build flavors are needed, please find instructions on building from source at [Build ONNX Runtime](BUILD.md). For production scenarios, it's strongly recommended to build from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

## Deploying ONNX Runtime
ONNX Runtime can be deployed to the cloud for model inferencing using [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service). See [detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) and [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx).

**ONNX Runtime Server (beta)** is a hosted application for serving ONNX models using ONNX Runtime, providing a REST API for prediction. Usage details can be found [here](https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md), and image installation instructions are [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#onnx-runtime-server-preview).

## Examples and Tutorials

## Getting ONNX Models
* The [ONNX Model Zoo](https://github.com/onnx/models) has popular ready-to-use pre-trained models.
* To export or convert a trained ONNX model trained from various frameworks, see [ONNX Tutorials](https://github.com/onnx/tutorials). Versioning comptability information can be found under [Versioning](docs/Versioning.md#tool-compatibility)
* Other services that can be used to create ONNX models include:
  * [Automated ML](aka.ms/automatedmldocs)
  * [Custom Vision](https://www.customvision.ai/)
  * [E2E training on Azure Machine Learning Services](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx)

## Why use ONNX Runtime
ONNX Runtime has an open architecture that is continually evolving to address the newest developments and challenges in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard, supporting all ONNX releases with future compatibility and maintaining backwards compatibility with prior releases.

ONNX Runtime continuously strives to provide top performance for a broad and growing number of usage scenarios in Machine Learning. Our investments focus on:
1. Run any ONNX model
2. High performance
3. Cross platform

## Run any ONNX model
**As of May 2019, ONNX Runtime supports up to ONNX 1.5 (opset10).**
ONNX Runtime provides comprehensive support of the ONNX spec and can be used to run all models based on ONNX v1.2.1 and higher. See version compatibility details [here](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md).

**Traditional ML support**
ONNX Runtime fully supports the [ONNX-ML profile](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) of the ONNX spec for traditional ML scenarios.

## High Performance
ONNX Runtime supports both CPU and GPU hardware. Using various graph optimizations and accelerators, ONNX Runtime can provide lower latency compared to other runtimes for faster end-to-end customer experiences and minimized machine utilization costs.

Currently ONNX Runtime supports the following accelerators:
* CPU
  * MLAS (Microsoft Linear Algebra Subprograms)
  * MKL-DNN
  * MKL-ML
  * Intel nGraph
* GPU
  * CUDA
  * TensorRT

Not all variations are supported in the [official release builds](#apis-and-official-builds), but can be built from source following the instructions [here](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md).

We are continuously working to integrate new execution providers for further improvements in latency and efficiency. If you are interested in contributing a new execution provider, please see [this page](docs/AddingExecutionProvider.md).

## Cross Platform
ONNX Runtime offers:
* APIs for Python, C#, and C
* Available for Linux, Windows, and Mac 

[API documentation and package installation](#Installation)

There are ongoing investments to make ONNX Runtime compatible with more platforms and architectures. If you have specific scenarios that are not currently supported, please share your suggestions and scenario details via [Github Issues](https://github.com/microsoft/onnxruntime/issues).

# Design and Key Features
* [High level architectural design](docs/HighLevelDesign.md)
* [Versioning](docs/Versioning.md)

## Extensibility Options
* [Add a custom operator/kernel](docs/AddingCustomOp.md)
* [Add an execution provider](docs/AddingExecutionProvider.md)
* [Add a new graph
transform](include/onnxruntime/core/optimizer/graph_transformer.h)
* [Add a new rewrite rule](include/onnxruntime/core/optimizer/rewrite_rule.h)

# Contribute
We welcome your contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

## Feedback
For any feedback or to report a bug, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License
[MIT License](LICENSE)
