<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is an open-source scoring engine for Open Neural Network Exchange (ONNX) models.

ONNX is an open format for machine learning (ML) models that is supported by various ML and DNN frameworks and tools. This format makes it easier to interoperate between frameworks and to maximize the reach of your hardware optimization investments. Learn more about ONNX on [https://onnx.ai](https://onnx.ai) or view the [Github Repo](https://github.com/onnx/onnx).

# Why use ONNX Runtime
ONNX Runtime has an open architecture that is continually evolving to address the newest developments and challenges in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard, supporting all ONNX releases with future compatibliity and maintaining backwards compatibility with prior releases.

ONNX Runtime continuously strives to provide top performance for a broad and growing number of usage scenarios in Machine Learning. Our investments focus on:
1. Run any ONNX model
2. High performance
3. Cross platform

## Run any ONNX model

### Alignment with ONNX Releases
ONNX Runtime provides comprehensive support of the ONNX spec and can be used to run all models based on ONNX v1.2.1 and higher. See ONNX version release details [here](https://github.com/onnx/onnx/releases).

As of May 2019, ONNX Runtime supports ONNX 1.5 (opset10). See [this table](https://github.com/Microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix) for details on ONNX Runtime and ONNX versioning compatibility, 

### Traditional ML support
ONNX Runtime fully supports the ONNX-ML profile of the ONNX spec for traditional ML scenarios.

## High Performance
ONNX Runtime supports both CPU and GPU hardware through a variety of execution providers. With a variety of graph optimizations and accelerators, ONNX Runtime often provides lower latency and higher efficiency compared to other runtimes. This provides faster end-to-end customer experiences and lower costs from improved machine utilization.

Currently ONNX Runtime supports CUDA, TensorRT, MLAS (Microsoft Linear Algebra Subprograms), MKL-DNN, MKL-ML, and nGraph for computation acceleration. See more details on available build options [here](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md).

We are continuously working to integrate new execution providers to provide improvements in latency and efficiency. If you are interested in contributing a new execution provider, please see [this page](docs/AddingExecutionProvider.md).

## Cross Platform
ONNX Runtime offers:
* APIs for Python, C#, and C
* Available for Linux, Windows, and Mac 

See API documentation and package installation instructions [below](#Installation).

We have ongoing investments to make ONNX Runtime compatible with more platforms and architectures. If you have specific scenarios that are not currently supported, please share your suggestions via [Github Issues](https://github.com/microsoft/onnxruntime/issues).

# Getting Started
ONNX models:
* Check out the [ONNX Model Zoo](https://github.com/onnx/models) for ready-to-use pre-trained models.
* To get an ONNX model by exporting from various frameworks, see [ONNX Tutorials](https://github.com/onnx/tutorials).

Once you have an ONNX model, you can [install the runtime](#Installation) for your machine to try it out. There is also an [ONNX-Ecosystem Docker container](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem) available and ready for use with the Python API.

One easy way to deploy the model on the cloud is by using [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning-service). See [detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx) and [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx). 


# Installation
## System Requirements
* ONNX Runtime binaries in CPU packages use OpenMP and depends on the library being available at runtime in the
system.
  * For Windows, OpenMP support comes as part of VC runtime. It is also available as redist packages:
    [vc_redist.x64.exe](https://aka.ms/vs/15/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/15/release/vc_redist.x86.exe)
  * For Linux, the system must have the libgomp.so.1 which can be installed using ```apt-get install libgomp1```.
* The official GPU builds require the CUDA 9.1 and cuDNN 7.1 runtime libraries being installed in the system.
* Python binaries are compatible with Python 3.5-3.7.
* Certain operators makes use of system locales. At the very least you will need to install English language package and configure en_US.UTF-8 locale.
  * For Ubuntu install language-pack-en package
  * Run the following commands:
    * locale-gen en_US.UTF-8
    * update-locale LANG=en_US.UTF-8
  * Follow similar procedure to configure other locales on other platforms.

## APIs and Official Builds
| API Documentation | CPU package | GPU package |
|-----|-------------|-------------|
| [Python](https://aka.ms/onnxruntime-python) | [Available on Pypi](https://pypi.org/project/onnxruntime)<br/><ul><li> Windows: x64</li><li>Linux: x64</li><li>Mac OS X: x64</li></ul><br/> | [Available on Pypi](https://pypi.org/project/onnxruntime-gpu) <br/><ul><li> Windows: x64</li><li>Linux: x64</li></ul><br/><br/> |
| [C#](docs/CSharp_API.md) | Available on Nuget : [MLAS+Eigen](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/), [MKL-ML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)</br><ul><li>Windows: x64</li><li>Linux: x64</li><li>Mac OS X: x64 (MLAS+Eigen only)</li></ul>| [Available on Nuget](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)<br/><ul><li> Windows: x64</li><li>Linux: x64</li></ul><br/>|
| [C](docs/C_API.md) | Available on Nuget : [MLAS+Eigen](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/), [MKL-ML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.MKLML/)</br><ul><li>Windows: x64</li><li>Linux: x64</li><li>Mac OS X: x64 (MLAS+Eigen only)</li></ul><br/>[Files (.zip, .tgz)](https://aka.ms/onnxruntime-release)<br/><ul><li>Windows: x64, x86</li><li>Linux: x64, x86</li><li>Mac OS X: x64 (MLAS+Eigen only)</li></ul> | [Available on Nuget](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)<br/><ul><li>Windows: x64</li><li>Linux: x64</li></ul><br/><br/>[Files (.zip, .tgz)](https://aka.ms/onnxruntime-release)<br/><ul><li>Windows: x64</li><li>Linux: x64</li></ul><br/> |
| [C++](onnxruntime/core/session/inference_session.h) | [Build from source](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md) | [Build from source](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md) |

For builds using other execution providers, see Build Details below.

## Build Details
For details on the build configurations and information on how to create a build, see [Build ONNX Runtime](BUILD.md).

## Versioning
See more details on API and ABI Versioning and ONNX Compatibility in [Versioning](docs/Versioning.md).

# Design and Key Features
For an overview of the high level architecture and key decisions in the technical design of ONNX Runtime, see [Engineering Design](docs/HighLevelDesign.md).

ONNX Runtime is built with an extensible design that makes it versatile to support a wide array of models with high performance.

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
