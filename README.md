# ONNX Runtime

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/onnxruntime%20CI%20Pipelines)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=1)

# Introduction 
ONNX Runtime is an open-source scoring engine for Open Neural Network Exchange (ONNX) models. 

ONNX is an open format for machine learning (ML) models that is supported by various ML and DNN frameworks and tools. This format makes it easier to interoperate between frameworks and to maximize the reach of your hardware optimization investments. Learn more about ONNX on [https://onnx.ai](https://onnx.ai) or view the [Github Repo](https://github.com/onnx/onnx). 
 
# Why use ONNX Runtime 
## Run any ONNX model
ONNX Runtime provides comprehensive support of the ONNX spec and can be used to run all models based on ONNX v1.2.1 and higher. See ONNX version release details [here](https://github.com/onnx/onnx/releases).

In order to support popular and leading AI models, the runtime stays up-to-date with evolving ONNX operators and functionalities. 
 
## Cross Platform 
ONNX Runtime offers:
* APIs for Python, C#, and C (experimental)
* Available for Linux, Windows, and Mac 

See API documentation and package installation instructions [below](#Installation). 
 
## High Performance 
You can use ONNX Runtime with both CPU and GPU hardware. You can also plug in additional execution providers to ONNX Runtime. With many graph optimizations and various accelerators, ONNX Runtime can often provide lower latency and higher efficiency compared to other runtimes. This provides smoother end-to-end customer experiences and lower costs from improved machine utilization.

Currently ONNX Runtime supports CUDA and MKL-DNN (with option to build with MKL) for computation acceleration, with more coming soon. To add an execution provider, please refer to [this page](docs/AddingExecutionProvider.md).
 
# Getting Started 
If you need a model:  
* Check out the [ONNX Model Zoo](https://github.com/onnx/models) for ready-to-use pre-trained models. 
* To get an ONNX model by exporting from various frameworks, see [ONNX Tutorials](https://github.com/onnx/tutorials).

If you already have an ONNX model, just [install the runtime](#Installation) for your machine to try it out. One easy way to deploy the model on the cloud is by using [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning-service). See detailed instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx). 

# Installation
## APIs and Official Builds
| API Documentation | CPU package | GPU package |
|-----|-------------|-------------|
| [Python](https://docs.microsoft.com/en-us/python/api/overview/azure/onnx/intro?view=azure-onnx-py) | [Windows](https://pypi.org/project/onnxruntime/)<br>[Linux](https://pypi.org/project/onnxruntime/)<br>[Mac](https://pypi.org/project/onnxruntime/)| [Windows](https://pypi.org/project/onnxruntime-gpu)<br>[Linux](https://pypi.org/project/onnxruntime-gpu/) |
| [C#](docs/CSharp_API.md) | [Windows](TODO)<br>Linux - Coming Soon<br>Mac - Coming Soon| Coming Soon |
| [C (experimental)](docs/C_API.md) | Coming Soon | Coming Soon |

There is also a non ABI [C++ API](onnxruntime/core/session/inference_session.h)

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
transform](include/onnxruntime/core/graph/graph_transformer.h)
* [Add a new rewrite rule](include/onnxruntime/core/graph/rewrite_rule.h)

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
