<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is a performance-focused inference engine for ONNX (Open Neural Network Exchange) models.

Models in the Tensorflow, Keras, PyTorch, scikit-learn, CoreML, and other popular [supported formats](https://github.com/onnx/tutorials#converting-to-onnx-format) can be converted to the standard [ONNX](https://onnx.ai) format, providing framework interoperability and helping to maximize the reach of hardware optimization investments. This provides a solution for systems to integrate a single inference engine to support models trained from a variety of frameworks, while taking advantage of specific hardware accelerators where available.

ONNX Runtime was designed with a focus on performance and scalability in order to support heavy workloads in high-scale production scenarios. It also has extensibility options for compatibility with emerging hardware developments.

ONNX Runtime stays up to date with the ONNX standard and supports all operators from the ONNX v1.2+ spec and is backwards compatible with older versions. Please refer to [this page](docs/Versioning.md) for ONNX opset compatibility details.



***
# Table of Contents
* **[Functional Overview](#functional-overview)**
  * [Key Features](#key-features)
  * [Performance Focused Design](#performance-focused-design)
    * [Performance Tuning](./docs/ONNX_Runtime_Perf_Tuning.md)
  * [Extensibility Options](#extensibility-options)
* **[Installation](#installation)**
  * [API Documentation](#api-documentation)
  * [Builds and Packages](#Builds-and-Packages)
* **[Usage](#usage)**
  * [Samples and Tutorials](./samples)
  * [Frequently Asked Questions](./docs/FAQ.md)
  * [Getting ONNX Models](#getting-onnx-models)
  * [Deploying ONNX Runtime](#deploying-onnx-runtime)
  * [Data/Telemetry](#Data/Telemetry)
* **[Contributions and Feedback](#contribute)**
* **[License](#license)**

***
# Functional Overview
## Key Features
* **Cross Platform:** The runtime provides a cross platform API compatible with Windows, Linux, and Mac and a variety of architectures. Both CPU and GPUs are supported, and language bindings are available for a variety of languages and architectures See more details ([below](apis-and-official-builds)). *If you have specific scenarios that are not supported, please share your suggestions and scenario details via [Github Issues](https://github.com/microsoft/onnxruntime/issues).*
* **Run any ONNX model:** ONNX Runtime provides comprehensive support of the ONNX spec and can be used to run all models based on ONNX v1.2.1 and higher. Both ONNX (DNN) and ONNX-ML (traditional ML) operator sets are supported. *The full set of operators and types supported is listed [here](./docs/OperatorKernels.md). Some operators not supported in the current ONNX version may be available as a [Contrib Operator](./docs/ContribOperators.md).*
* **Backwards Compatible**: Newer versions of ONNX Runtime support all models that worked with prior versions, so updates should not break integrations. *See version compatibility details [here](./docs/Versioning.md).*


## Performance Focused Design
 [High level architectural design](docs/HighLevelDesign.md)
 
Using various graph optimizations and accelerators, ONNX Runtime can provide lower latency compared to other runtimes for faster end-to-end customer experiences and minimized machine utilization costs. See  [Performance Tuning guidance](./docs/ONNX_Runtime_Perf_Tuning.md).

### Supported Accelerators
The list of currently supported accelerators (termed [Execution Providers](./docs/execution_providers)) is below. Please see [BUILD.md](./BUILD.md) for build instructions. If you are interested in contributing a new execution provider, please see [this page](docs/AddingExecutionProvider.md).

Please refer to [Roadmap](./docs/Roadmap.md#accelerators-and-execution-providers) for a list of upcoming accelerators. 

#### CPU
* Default CPU - *MLAS (Microsoft Linear Algebra Subprograms) + Eigen*
* [Intel DNNL](./docs/execution_providers/DNNL-ExecutionProvider.md)
* [Intel nGraph](./docs/execution_providers/nGraph-ExecutionProvider.md)
* Intel MKL-ML 

#### GPU
* NVIDIA CUDA
* [NVIDIA TensorRT](./docs/execution_providers/TensorRT-ExecutionProvider.md)
* [DirectML](./docs/execution_providers/DirectML-ExecutionProvider.md)

#### IoT/Edge/Mobile
* [Intel OpenVINO](./docs/execution_providers/OpenVINO-ExecutionProvider.md)
* [ARM Compute Library](./docs/execution_providers/ACL-ExecutionProvider.md) (*preview*)
* [Android Neural Networks API](./docs/execution_providers/NNAPI-ExecutionProvider.md) (*preview*)

#### Other
* [Nuphar Model Compiler](./docs/execution_providers/Nuphar-ExecutionProvider.md)

## Extensibility Options
  * [Add a custom operator/kernel](docs/AddingCustomOp.md)
  * [Add a new graph transform](include/onnxruntime/core/optimizer/graph_transformer.h)
  * [Add a new rewrite rule](include/onnxruntime/core/optimizer/rewrite_rule.h)
  * [Add an execution provider](docs/AddingExecutionProvider.md)

***

# Installation
**Quick Start:** The [ONNX-Ecosystem Docker container image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem) is available on Dockerhub and includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started. Additional dockerfiles can be found [here](./dockerfiles).

## API Documentation

|Language|Supported Versions|Samples|
|--|--|--|
[Python](https://aka.ms/onnxruntime-python)| 3.5, 3.6, 3.7<br>[Python Dev Notes](./docs/Python_Dev_Notes.md)| [Samples](./samples#python)|
|[C#](docs/CSharp_API.md)| | [Samples](./samples#C)|
|[C++](./include/onnxruntime/core/session/onnxruntime_cxx_api.h)| |[Samples](./samples#CC)|
|[C](docs/C_API.md)| | [Samples](./samples#CC)|
|[WinRT](docs/WinRT_API.md) | [Windows.AI.MachineLearning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)| [Samples](https://github.com/microsoft/windows-Machine-Learning)|
|[Java](docs/Java_API.md)|8-13|[Samples](./samples#Java)| 
[Ruby](https://github.com/ankane/onnxruntime) (external project)| 2.4-2.7| [Samples](https://ankane.org/tensorflow-ruby)|

## Builds and Packages

Official builds are available for:
* Default CPU Provider (Eigen + MLAS)
* GPU Provider - NVIDIA CUDA
  * *note: If your deployment target is Windows, the [DirectML execution provider](./docs/execution_providers/DirectML-ExecutionProvider.md) is recommended for optimal performance and compatibility with a broad set of GPUs. This will be an official package soon. In the meantime, see the build instructions at [BUILD.md](./BUILD.md#directml).* 

Python packages can be found on PyPi, and C#/C/C++ packages on Nuget. Please view the table on [aka.ms/onnxruntime](https://aka.ms/onnxruntime) for instructions for different build combinations. 

For additional build flavors and/or dockerfiles, please carefully read through [BUILD.md](./BUILD.md). If you encounter problems, please provide as much information as possible when filing an [issue](https://github.com/Microsoft/onnxruntime/issues). 

For production scenarios, it's strongly recommended to build only from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

#### PyPi (Python):
*If using `pip` to download the Python binaries, run `pip install --upgrade pip` prior to downloading.*

* [onnxruntime](https://pypi.org/project/onnxruntime)
* [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)

#### Nuget (C#/C/C++):
* [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime)
* [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)

#### Other package repositories:
Contributed non-official packages (including Homebrew, Linuxbrew, and nixpkgs) are listed [here](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp). These are not maintained by the core ONNX Runtime team and will have limited support; use at your discretion.

### System Requirements
These system requirements must be met for using the compiled binaries. 

#### System language
* Installation of the **English language package** and configuring `en_US.UTF-8 locale` is required, as certain operators makes use of system locales. 
* For Ubuntu, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
  * Run the following commands:
    `locale-gen en_US.UTF-8`
    `update-locale LANG=en_US.UTF-8`
  * Follow similar procedure to configure other locales on other platforms.
  
#### Default CPU
* ONNX Runtime binaries in the CPU packages use OpenMP and depend on the library being available at runtime in the
system.
  * For Windows, **OpenMP** support comes as part of VC runtime. It is also available as redist packages:
    [vc_redist.x64.exe](https://aka.ms/vs/16/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/16/release/vc_redist.x86.exe)
  * For Linux, the system must have **libgomp.so.1** which can be installed using `apt-get install libgomp1`.

#### Default GPU (CUDA)
* The default GPU build requires CUDA runtime libraries being installed on the system:
	 * Version: **CUDA 10.1** and **cuDNN 7.6.5**
* Version dependencies from older ONNX Runtime releases can be found in [prior release notes](https://github.com/microsoft/onnxruntime/releases).

#### Other Execution Providers
* For requirements and dependencies of other build options, see detailed build instructions on the [BUILD.md](./BUILD.md#additional-build-instructions) page.
***
# Usage
## [Samples and Tutorials](./samples)

## [Frequently Asked Questions](./docs/FAQ.md)

## Getting ONNX Models
To get an ONNX model, please view these [ONNX Tutorials](https://github.com/onnx/tutorials#getting-onnx-models).
ONNX Runtime supports all versions of ONNX 1.2+. Full versioning compatibility information can be found under [Versioning](docs/Versioning.md#tool-compatibility).

## Deploying ONNX Runtime
### Cloud
ONNX Runtime can be deployed to the cloud for model inferencing using [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service). See [detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) and [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx).

**ONNX Runtime Server (beta)** is a hosted application for serving ONNX models using ONNX Runtime, providing a REST API for prediction. Usage details can be found [here](./docs/ONNX_Runtime_Server_Usage.md), and image installation instructions are [here](./dockerfiles#onnx-runtime-server-preview).

### IoT and edge devices
The expanding focus and selection of IoT devices with sensors and consistent signal streams introduces new opportunities to move AI workloads to the edge.
 
This is particularly important when there are massive volumes of incoming data/signals that may not be efficient or useful to push to the cloud due to storage or latency considerations. Consider: surveillance tapes where 99% of footage is uneventful, or real-time person detection scenarios where immediate action is required. In these scenarios, directly executing model inferencing on the target device is crucial for optimal assistance.
 
To deploy AI workloads to these edge devices and take advantage of hardware acceleration capabilities on the target device, see [these reference implementations](https://github.com/Azure-Samples/onnxruntime-iot-edge).

### Client applications
Install or build the package you need to use in your application. Check [this page](https://microsoft.github.io/onnxruntime/) for installation/package guidance. See [sample implementations](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx) using the C++ API. 

On newer Windows 10 devices (1809+), ONNX Runtime is available by default as part of the OS and is accessible via the [Windows Machine Learning APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/). Find tutorials [here](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop) for building a Windows Desktop or UWP application using WinML.

## Data/Telemetry
This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.
***
# Contribute
We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

## Feedback
For any feedback or to report a bug, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

***
# License
[MIT License](LICENSE)
