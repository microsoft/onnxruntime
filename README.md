<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is a cross-platform **inferencing and training accelerator** compatible with many popular ML/DNN frameworks, including PyTorch, TensorFlow/Keras, scikit-learn, and more. **[aka.ms/onnxruntime](https://aka.ms/onnxruntime)**


Many users can benefit from ONNX Runtime, including those looking to:
* Improve inference performance for a wide variety of ML models
* Reduce time and cost of training large models
* Train in Python but deploy into a C#/C++/Java app
* Run on different hardware and operating systems
* Support models created in several different frameworks

[ONNX Runtime inferencing](./onnxruntime) APIs are stable and production-ready since the [1.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.0.0) in October 2019 and can enable faster customer experiences and lower costs.

[ONNX Runtime training](./orttraining) feature was introduced in May 2020 in preview. This feature supports acceleration of PyTorch training on multi-node NVIDIA GPUs for transformer models. Additional updates for this feature are coming soon.


***

# Table of Contents

* **[Get Started](#get-started)**
  * [ONNX Runtime Inferencing](#inferencing-start)
  * [ONNX Runtime Training](#training-start)
* **[Data/Telemetry](#DataTelemetry)**
* **[Contributions and Feedback](#contributions-and-feedback)**
* **[License](#license)**

***

# Get Started

[Frequently Asked Questions](./docs/FAQ.md)

## Inferencing: Start

To use ONNX Runtime, refer to the table on [aka.ms/onnxruntime](https://aka.ms/onnxruntime) for instructions for different build combinations. 

* [Compatibility](#compatibility)
* [Binaries](#binaries)
* [Build from source (includes additional combinations)](#build-from-source)
* [Docker images](#docker-images)
* [API documentation](#api-documentation)
* [Hardware accelerators](#supported-accelerators)
* [Deploy ONNX Runtime inferencing](#deploying-onnx-runtime)
* [Samples](./samples)
* [High level architectural design](docs/InferenceHighLevelDesign.md)
* [Performance Tuning](./docs/ONNX_Runtime_Perf_Tuning.md)
* [Extensibility: Add a new graph transform](include/onnxruntime/core/optimizer/graph_transformer.h)
* [Extensibility: Add a new rewrite rule](include/onnxruntime/core/optimizer/rewrite_rule.h)

### Compatibility

Supporting models based on the standard [ONNX](https://onnx.ai) format, the runtime is compatible with PyTorch, scikit-learn, TensorFlow, Keras, and all other frameworks and tools that support the interoperable format.

* [Getting ONNX models - tutorials](https://github.com/onnx/tutorials#getting-onnx-models)

ONNX Runtime is up to date and backwards compatible with all operators (both DNN and traditional ML) since ONNX v1.2.1+. [(ONNX compatibility details)](docs/Versioning.md). Newer versions of ONNX Runtime support all models that worked with prior versions, so updates should not break integrations. 

* [Supported operators/types](./docs/OperatorKernels.md)
  * *Operators not supported in the current ONNX spec may be available as a [Contrib Operator](./docs/ContribOperators.md)*
* [Extensibility: Add a custom operator/kernel](docs/AddingCustomOp.md)

### Binaries

Official builds are available on PyPi (Python) and Nuget (C#/C/C++):

* Default CPU Provider (Eigen + MLAS)
* GPU Provider - NVIDIA CUDA
* GPU Provider - DirectML (Windows)
  * *On Windows, the [DirectML execution provider](./docs/execution_providers/DirectML-ExecutionProvider.md) is recommended for optimal performance and compatibility with a broad set of GPUs.*

Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.

|Pypi (Python)|Nuget (C#/C/C++)|Other package repositories|
|---|---|---|
*If using pip, run `pip install --upgrade pip` prior to downloading.*<br><br>CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime) / [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)<br><br>GPU: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) / [ort-gpu-nightly (dev)](https://test.pypi.org/project/ort-gpu-nightly) | CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) / [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) <br><br>GPU: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) / [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)</li></ul>|[Contributed non-official packages](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp) (including Homebrew, Linuxbrew, and nixpkgs)<br><br>*These are not maintained by the core ONNX Runtime team and may have limited support; use at your discretion.*|

#### System Requirements

The following are required for usage of the official published packages.

* Visual C++ Runtime (for Windows packages)
  * Requires [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
* System language 
  * Installation of the **English language package** and configuring `en_US.UTF-8 locale` is required, as certain operators makes use of system locales. 
  * For Ubuntu, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
    * Run the following commands:
      `locale-gen en_US.UTF-8`
      `update-locale LANG=en_US.UTF-8`
    * Follow similar procedure to configure other locales on other platforms.
  
* Default CPU
  * ONNX Runtime binaries in the CPU packages use OpenMP and depend on the library being available at runtime in the system.
    * For Windows, **OpenMP** support comes as part of VC runtime. It is also available as redist packages:
      [vc_redist.x64.exe](https://aka.ms/vs/16/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/16/release/vc_redist.x86.exe)
    * For Linux, the system must have **libgomp.so.1** which can be installed using `apt-get install libgomp1`.

* Default GPU (CUDA)
  * The default GPU build requires CUDA runtime libraries being installed on the system:
    * Version: **CUDA 10.1** and **cuDNN 7.6.5**
  * Version dependencies from older ONNX Runtime releases can be found in [prior release notes](https://github.com/microsoft/onnxruntime/releases).

### Build from Source

For production scenarios, it's strongly recommended to build only from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

* [Instructions for additional build flavors](./BUILD.md)

### Docker Images

* [ONNX-Ecosystem](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem): includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started
* [Additional dockerfiles](./dockerfiles)

### API Documentation

|API|Supported Versions|Samples|
|---|---|---|
[Python](https://aka.ms/onnxruntime-python)| 3.5, 3.6, 3.7<br>[Python Dev Notes](./docs/Python_Dev_Notes.md)| [Samples](./samples#python)|
|[C#](docs/CSharp_API.md)| | [Samples](./samples#C)|
|[C++](./include/onnxruntime/core/session/onnxruntime_cxx_api.h)| |[Samples](./samples#CC)|
|[C](docs/C_API.md)| | [Samples](./samples#CC)|
|[WinRT](docs/WinRT_API.md) | [Windows.AI.MachineLearning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)| [Samples](https://github.com/microsoft/windows-Machine-Learning)|
|[Java](docs/Java_API.md)|8-13|[Samples](./samples#Java)| 
[Ruby](https://github.com/ankane/onnxruntime) (external project)| 2.4-2.7| [Samples](https://ankane.org/tensorflow-ruby)|
|[Javascript (node.js)](./nodejs) |12.x | [Samples](./nodejs/examples/README.md) |

### Supported Accelerators

[Execution Providers](./docs/execution_providers)

|CPU|GPU|IoT/Edge/Mobile|Other|
|---|---|---|---|
|<ul><li>Default CPU - *MLAS (Microsoft Linear Algebra Subprograms) + Eigen*</li><li>[Intel DNNL](./docs/execution_providers/DNNL-ExecutionProvider.md)</li><li>[Intel nGraph](./docs/execution_providers/nGraph-ExecutionProvider.md)</li><li>Intel MKL-ML *(build option)*</li></ul>|<ul><li>NVIDIA CUDA</li><li>[NVIDIA TensorRT](./docs/execution_providers/TensorRT-ExecutionProvider.md)</li><li>[DirectML](./docs/execution_providers/DirectML-ExecutionProvider.md)</li><li>[AMD MIGraphX](./docs/execution_providers/MIGraphX-ExecutionProvider.md)</li></ul>|<ul><li>[Intel OpenVINO](./docs/execution_providers/OpenVINO-ExecutionProvider.md)</li><li>[ARM Compute Library](./docs/execution_providers/ACL-ExecutionProvider.md) (*preview*)</li><li>[Android Neural Networks API](./docs/execution_providers/NNAPI-ExecutionProvider.md) (*preview*)</li></ul>|<ul><li>[Nuphar Model Compiler](./docs/execution_providers/Nuphar-ExecutionProvider.md) - (*preview*)</li><li>[Rockchip NPU](./docs/execution_providers/RKNPU-ExecutionProvider.md) (*preview*)</li><li>[Xilinx Vitis-AI](./docs/execution_providers/Vitis-AI-ExecutionProvider.md) (*preview*)</li></ul>| 

* [Roadmap: Upcoming accelerators](./docs/Roadmap.md#accelerators-and-execution-providers)
* [Extensibility: Add an execution provider](docs/AddingExecutionProvider.md)

### Deploying ONNX Runtime

#### Cloud

* ONNX Runtime can be deployed to any cloud for model inferencing, including [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service).
  * [Detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx)
  * [AzureML sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)

* **ONNX Runtime Server (beta)** is a hosting application for serving ONNX models using ONNX Runtime, providing a REST API for prediction.
  * [Usage details](./docs/ONNX_Runtime_Server_Usage.md)
  * [Image installation instructions](./dockerfiles#onnx-runtime-server-preview)

#### IoT and edge devices

* [Reference implementations](https://github.com/Azure-Samples/onnxruntime-iot-edge)

The expanding focus and selection of IoT devices with sensors and consistent signal streams introduces new opportunities to move AI workloads to the edge.
This is particularly important when there are massive volumes of incoming data/signals that may not be efficient or useful to push to the cloud due to storage or latency considerations. Consider: surveillance tapes where 99% of footage is uneventful, or real-time person detection scenarios where immediate action is required. In these scenarios, directly executing model inferencing on the target device is crucial for optimal assistance.

#### Client applications

* Install or build the package you need to use in your application. ([sample implementations](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx) using the C++ API)

* On newer Windows 10 devices (1809+), ONNX Runtime is available by default as part of the OS and is accessible via the [Windows Machine Learning APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/). ([Tutorials for Windows Desktop or UWP app](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop))

***

## Training: Start

The ONNX Runtime training feature enables easy integration with existing Pytorch trainer code to accelerate the exection. With a few lines of code, you can add ONNX Runtime into your existing training scripts and start seeing acceleration. The current preview version supports training acceleration for transformer models on NVIDIA GPUs.

**[ONNX Runtime pre-training sample](https://github.com/microsoft/onnxruntime-training-examples)**: This sample is setup to pre-train the BERT-Large model to show how ONNX Runtime training can be used to accelerate training execution.

### Train PyTorch model with ONNX Runtime
ONNX Runtime (ORT) has the capability to train existing PyTorch models through its optimized backend. For this, we have introduced an python API for PyTorch, called ORTTrainer, which can be used to switch the training backend for PyTorch models (instance of `torch.nn.Module`) to `orttrainer`. This requires some changes in the trainer code, such as replacing the PyTorch optimizer, and optionally, setting flags to enable additional features such as mixed-precision training. Here is a sample code fragment to integrate ONNX Runtime Training in your PyTorch pre-training script:

_NOTE: The current API is experimental and expected to see significant changes in the near future. Our goal is to improve the interface to provide a seamless integration with PyTorch training that requires minimal changes in users’ training code._ 

  ```python
  import torch
  ...
  import onnxruntime
  from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer

  # Model definition
  class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
      ...
    def forward(self, x):
      ...

  model = Net(D_in, H, H_out)
  criterion = torch.nn.Functional.cross_entropy
  description = ModelDescription(...)
  optimizer = 'SGDOptimizer'
  trainer = ORTTrainer(model, criterion, description, optimizer, ...)

  # Training Loop
  for t in range(1000):
    # forward + backward + weight update
    loss, y_pred = trainer.train_step(x, y, learning_rate)
    ...
  ```

### Build ONNX Runtime Training from source
To use ONNX Runtime training in a custom environment, like on-prem NVIDIA DGX-2 clusters, you can use these [build instructions](BUILD.md#training) to generate the Python package to integrate into existing trainer code.



# Data/Telemetry

This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

# Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For any feedback or to report a bug, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

This project is licensed under the [MIT License](LICENSE).
