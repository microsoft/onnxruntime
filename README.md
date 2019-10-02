<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is a performance-focused complete scoring engine for Open Neural Network Exchange (ONNX) models, with an open extensible architecture to continually address the latest developments in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard with complete implementation of **all** ONNX operators, and supports all ONNX releases (1.2+) with both future and backwards compatibility. Please refer to [this page](docs/Versioning.md) for ONNX opset compatibility details.

[ONNX](https://onnx.ai) is an interoperable format for machine learning models supported by various ML and DNN frameworks and tools. The universal format makes it easier to interoperate between frameworks and maximize the reach of hardware optimization investments.

***
**[Key Features](#key-features)**

**Setup**
* [Installation](#installation)
* [APIs and Official Binaries](#apis-and-official-builds)
* [Building from Source](#building-from-source)

**Usage**
* [Getting ONNX Models](#getting-onnx-models)
* [Deploying ONNX Runtime](#deploying-onnx-runtime)
* [Performance Tuning](#performance-tuning)

**[Examples and Tutorials](#examples-and-tutorials)**

**More Info**
* [Technical Design Details](#technical-design-details)
* [Extensibility Options](#extensibility-options)

**[Contributions and Feedback](#contribute)**

**[License](#license)**
***
# Key Features
## Run any ONNX model
ONNX Runtime provides comprehensive support of the ONNX spec and can be used to run all models based on ONNX v1.2.1 and higher. See version compatibility details [here](./docs/Versioning.md).

**Traditional ML support**

In addition to DNN models, ONNX Runtime fully supports the [ONNX-ML profile](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md) of the ONNX spec for traditional ML scenarios.

For the full set of operators and types supported, please see [operator documentation](./docs/OperatorKernels.md)

*Note: Some operators not supported in the current ONNX version may be available as a [Contrib Operator](./docs/ContribOperators.md)*


## High Performance
ONNX Runtime supports both CPU and GPU. Using various graph optimizations and accelerators, ONNX Runtime can provide lower latency compared to other runtimes for faster end-to-end customer experiences and minimized machine utilization costs.

Currently ONNX Runtime supports the following accelerators:
* MLAS (Microsoft Linear Algebra Subprograms)
* [MKL-DNN](./docs/execution_providers/MKL-DNN-ExecutionProvider.md) - [subgraph optimization](./docs/execution_providers/MKL-DNN-Subgraphs.md)
* MKL-ML
* [Intel nGraph](./docs/execution_providers/nGraph-ExecutionProvider.md)
* CUDA
* [TensorRT](./docs/execution_providers/TensorRT-ExecutionProvider.md)
* [OpenVINO](./docs/execution_providers/OpenVINO-ExecutionProvider.md)
* [Nuphar](./docs/execution_providers/Nuphar-ExecutionProvider.md)

Not all variations are supported in the [official release builds](#apis-and-official-builds), but can be built from source following [these instructions](./BUILD.md). Find Dockerfiles [here](./dockerfiles).

We are continuously working to integrate new execution providers for further improvements in latency and efficiency. If you are interested in contributing a new execution provider, please see [this page](docs/AddingExecutionProvider.md).


## Cross Platform
[API documentation and package installation](#installation)

ONNX Runtime is currently available for Linux, Windows, and Mac with Python, C#, C++, and C APIs.
If you have specific scenarios that are not supported, please share your suggestions and scenario details via [Github Issues](https://github.com/microsoft/onnxruntime/issues).
***
# Installation
**Quick Start:** The [ONNX-Ecosystem Docker container image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem) is available on Dockerhub and includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started.

Additional dockerfiles for some features can be found [here](./dockerfiles).

## APIs and Official Builds

### API Documentation
* [Python](https://aka.ms/onnxruntime-python)
* [C](docs/C_API.md)
* [C#](docs/CSharp_API.md)
* [C++](./include/onnxruntime/core/session/onnxruntime_cxx_api.h)

### Official Builds
| | CPU (MLAS+Eigen) | CPU (MKL-ML) | GPU (CUDA)
|---|:---|:---|:---|
| **Python** | **[pypi: onnxruntime](https://pypi.org/project/onnxruntime)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | -- | **[pypi: onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)**<br><br>Windows (x64)<br>Linux (x64) |
| **C#** | **[Nuget: Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/)**<br><br>Windows (x64, x86)<br>Linux (x64, x86)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.MKLML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)**<br><br>Windows (x64)<br>Linux (x64) |
| **C/C++ wrapper** | **[Nuget: Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime)**<br><br>**[.zip, .tgz](https://aka.ms/onnxruntime-release)**<br><br>Windows (x64, x86)<br>Linux (x64, x86)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.MKLML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)**<br><br>Windows (x64)<br>Linux (x64)<br>Mac OS X (x64) | **[Nuget: Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)**<br><br>**[.zip, .tgz](https://aka.ms/onnxruntime-release)**<br><br>Windows (x64)<br>Linux (x64) |

#### System Requirements (pre-requisite dependencies)
* ONNX Runtime binaries in the CPU packages use OpenMP and depend on the library being available at runtime in the
system.
  * For Windows, **OpenMP** support comes as part of VC runtime. It is also available as redist packages:
    [vc_redist.x64.exe](https://aka.ms/vs/15/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/15/release/vc_redist.x86.exe)
  * For Linux, the system must have **libgomp.so.1** which can be installed using `apt-get install libgomp1`.
* GPU builds require CUDA runtime libraries being installed on the system:
	 * Version: **CUDA 10.0** and **cuDNN 7.3**
	 * Linux Python packages require **CUDA 10.1** and **cuDNN 7.6** 
  * Older ONNX Runtime releases: used **CUDA 9.1** and **cuDNN 7.1** - please refer to [prior release notes](https://github.com/microsoft/onnxruntime/releases) for more details.
* Python binaries are compatible with **Python 3.5-3.7**. See [Python Dev Notes](./docs/Python_Dev_Notes.md). If using `pip` to be download the Python binaries, run `pip install --upgrade pip` prior to downloading. 
* Certain operators makes use of system locales. Installation of the **English language package** and configuring `en_US.UTF-8 locale` is required.
  * For Ubuntu install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
  * Run the following commands:
    `locale-gen en_US.UTF-8`
    `update-locale LANG=en_US.UTF-8`
  * Follow similar procedure to configure other locales on other platforms.

## Building from Source
If additional build flavors are needed, please find instructions on building from source at [Build ONNX Runtime](BUILD.md). For production scenarios, it's strongly recommended to build from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

Dockerfiles are available [here](./tools/ci_build/github/linux/docker) to help you get started.

***
# Usage

## Getting ONNX Models
* The [ONNX Model Zoo](https://github.com/onnx/models) has popular ready-to-use pre-trained models.
* To export or convert a trained ONNX model trained from various frameworks, see [ONNX Tutorials](https://github.com/onnx/tutorials). Versioning compatibility information can be found under [Versioning](docs/Versioning.md#tool-compatibility)
* Other services that can be used to create ONNX models include:
  * [AutoML from AzureML SDK](https://aka.ms/automatedmldocs)
  * [Custom Vision](https://www.customvision.ai/)
  * [E2E training on Azure Machine Learning Services](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx)

## Deploying ONNX Runtime
ONNX Runtime can be deployed to the cloud for model inferencing using [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service). See [detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) and [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx).

**ONNX Runtime Server (beta)** is a hosted application for serving ONNX models using ONNX Runtime, providing a REST API for prediction. Usage details can be found [here](./docs/ONNX_Runtime_Server_Usage.md), and image installation instructions are [here](./dockerfiles#onnx-runtime-server-preview).

## Performance Tuning
ONNX Runtime is open and extensible, supporting a broad set of configurations and execution providers for model acceleration. For performance tuning guidance, please see [this page](./docs/ONNX_Runtime_Perf_Tuning.md).

To tune performance for ONNX models, the [ONNX Go Live tool "OLive"](https://github.com/microsoft/OLive) provides an easy-to-use pipeline for converting models to ONNX and optimizing performance for inferencing with ONNX Runtime. 

***
# Examples and Tutorials
## Python
**Inference only**
* [Model Inferencing (single node Sigmoid)](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb)
* [Model Inferencing (Resnet50)](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
* [Model Inferencing](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem/inference_demos) using [ONNX-Ecosystem Docker image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)
* [Model Inferencing using ONNX Runtime Server (SSD Single Shot MultiBox Detector)](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb)

**Inference with model conversion**
* [SKL Pipeline: Train, Convert, and Inference](https://microsoft.github.io/onnxruntime/auto_examples/plot_train_convert_predict.html#sphx-glr-auto-examples-plot-train-convert-predict-py)
* [Keras: Convert and Inference](https://microsoft.github.io/onnxruntime/auto_examples/plot_dl_keras.html#sphx-glr-auto-examples-plot-dl-keras-py)

**Inference and deploy through AzureML**
* Inferencing using [ONNX Model Zoo](https://github.com/onnx/models) models: 
  * [Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb) 
  * [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
  * [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
* Convert existing model for Inferencing:
  * [TinyYolo](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb)
* Train a model with PyTorch and Inferencing:
  * [MNIST](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-train-pytorch-aml-deploy-mnist.ipynb)

* GPU: Inferencing with TensorRT Execution Provider (AKS)
  * [FER+](./docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb)

**Inference and Deploy wtih Azure IoT Edge**
  * [Intel OpenVINO](http://aka.ms/onnxruntime-openvino)
  * [NVIDIA TensorRT on Jetson Nano (ARM64)](http://aka.ms/onnxruntime-arm64)

**Other**
* [Running ONNX model tests](./docs/Model_Test.md)


## C#
* [Inferencing Tutorial](./docs/CSharp_API.md#getting-started)


## C/C++
* [C - Inferencing (SqueezeNet)](./csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)
* [C++ - Inferencing (SqueezeNet)](./csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp)
* [C++ - Inferencing (MNIST)](./samples/c_cxx/MNIST)

***
# Technical Design Details
* [High level architectural design](docs/HighLevelDesign.md)
* [Versioning](docs/Versioning.md)

## Extensibility Options
* [Add a custom operator/kernel](docs/AddingCustomOp.md)
* [Add an execution provider](docs/AddingExecutionProvider.md)
* [Add a new graph
transform](include/onnxruntime/core/optimizer/graph_transformer.h)
* [Add a new rewrite rule](include/onnxruntime/core/optimizer/rewrite_rule.h)

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
