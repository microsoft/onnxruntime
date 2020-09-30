---
title: Get Started
has_children: true
nav_order: 1
---

# Get Started
{: .no_toc }

ONNX Runtime is a cross-platform accelerator for training and inference, compatible with PyTorch, TensorFlow/Keras, scikit-learn, and more.

You can benefit from ONNX Runtime if you want to:

* Improve inference performance of machine learning models
* Reduce time and cost of training large models
* Train in Python but deploy into a C#/C++/Java app
* Run on different hardware and operating systems
* Support models created in several different frameworks

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Use ONNX Runtime for Inference

### Docker Images

* [ONNX-Ecosystem](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem): includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started
* [Additional dockerfiles](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)

### API Documentation

|API|Supported Versions|Samples|
|---|---|---|
[Python](https://aka.ms/onnxruntime-python)| 3.5, 3.6, 3.7<br>[Python Dev Notes](https://github.com/microsoft/onnxruntime/tree/master/docs/Python_Dev_Notes.md)| [Samples](https://github.com/microsoft/onnxruntime/tree/master/samples/#python)|
|[C#](../reference/api/csharp-api.md)| | [Samples](https://github.com/microsoft/onnxruntime/tree/master/samples/#C)|
|[C++](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h)| |[Samples](https://github.com/microsoft/onnxruntime/tree/master/samples/#CC)|
|[C](../reference/api/c-api.md)| | [Samples](https://github.com/microsoft/onnxruntime/tree/master/samples/#CC)|
|[WinRT](../reference/api/winrt-api.md) | [Windows.AI.MachineLearning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)| [Samples](https://github.com/microsoft/windows-Machine-Learning)|
|[Java](../reference/api/java-api.md)|8-13|[Samples](https://github.com/microsoft/onnxruntime/tree/master/samples/#Java)| 
[Ruby](https://github.com/ankane/onnxruntime) (external project)| 2.4-2.7| [Samples](https://ankane.org/tensorflow-ruby)|
|[Javascript (node.js)](../reference/api/nodejs-api.md) |12.x | [Samples](https://github.com/microsoft/onnxruntime/blob/master/samples/nodejs) |

### Supported Accelerators

[Execution Providers](../reference/execution-providers)

|CPU|GPU|IoT/Edge/Mobile|Other|
|---|---|---|---|
|Default CPU - *MLAS (Microsoft Linear Algebra Subprograms) + Eigen*|NVIDIA CUDA|[Intel OpenVINO](../reference/execution-providers/OpenVINO-ExecutionProvider.md)|[Nuphar Model Compiler](../reference/execution-providers/Nuphar-ExecutionProvider.md) (*preview*)|
|[Intel DNNL](../reference/execution-providers/DNNL-ExecutionProvider.md)|[NVIDIA TensorRT](../reference/execution-providers/TensorRT-ExecutionProvider.md)|[ARM Compute Library](../reference/execution-providers/ACL-ExecutionProvider.md) (*preview*)|[Rockchip NPU](../reference/execution-providers/RKNPU-ExecutionProvider.md) (*preview*)|
|[Intel nGraph](../reference/execution-providers/nGraph-ExecutionProvider.md)|[DirectML](../reference/execution-providers/DirectML-ExecutionProvider.md)|[Android Neural Networks API](../reference/execution-providers/NNAPI-ExecutionProvider.md) (*preview*)|[Xilinx Vitis-AI](../reference/execution-providers/Vitis-AI-ExecutionProvider.md) (*preview*)|
|Intel MKL-ML *(build option)*|[AMD MIGraphX](../reference/execution-providers/MIGraphX-ExecutionProvider.md)|[ARM-NN](../reference/execution-providers/ArmNN-ExecutionProvider.md) (*preview*)|

* [Roadmap: Upcoming accelerators](https://github.com/microsoft/onnxruntime/tree/master/docs/Roadmap.md#accelerators-and-execution-providers)

### Deploying ONNX Runtime

#### Cloud

* ONNX Runtime can be deployed to any cloud for model inference, including [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service).
  * [Detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx)
  * [AzureML sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)

* **ONNX Runtime Server (beta)** is a hosting application for serving ONNX models using ONNX Runtime, providing a REST API for prediction.
  * [Usage details](https://github.com/microsoft/onnxruntime/tree/master/docs/ONNX_Runtime_Server_Usage.md)
  * [Image installation instructions](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#onnx-runtime-server-preview)

#### IoT and edge devices

* [Reference implementations](https://github.com/Azure-Samples/onnxruntime-iot-edge)

The expanding focus and selection of IoT devices with sensors and consistent signal streams introduces new opportunities to move AI workloads to the edge.
This is particularly important when there are massive volumes of incoming data/signals that may not be efficient or useful to push to the cloud due to storage or latency considerations. Consider: surveillance tapes where 99% of footage is uneventful, or real-time person detection scenarios where immediate action is required. In these scenarios, directly executing model inference on the target device is crucial for optimal assistance.

#### Client applications

* Install or build the package you need to use in your application. ([sample implementations](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx) using the C++ API)

* On newer Windows 10 devices (1809+), ONNX Runtime is available by default as part of the OS and is accessible via the [Windows Machine Learning APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/). ([Tutorials for Windows Desktop or UWP app](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop))

### Build from Source

For production scenarios, it's strongly recommended to build only from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

* [Instructions for additional build flavors](../how-to/build.md)

***

## Use ONNX Runtime for Training

The ONNX Runtime training feature enables easy integration with existing Pytorch trainer code to accelerate the execution. With a few lines of code, you can add ONNX Runtime into your existing training scripts and start seeing acceleration. The current preview version supports training acceleration for transformer models on NVIDIA GPUs.

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

To use ONNX Runtime training in a custom environment, like on-prem NVIDIA DGX-2 clusters, you can use these [build instructions](../how-to/build.md#training) to generate the Python package to integrate into existing trainer code.

