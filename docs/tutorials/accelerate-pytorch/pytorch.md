---
title: PyTorch Inference
description: How to run PyTorch models efficiently and across multiple platforms
parent: Accelerate PyTorch
grand_parent: Tutorials
nav_order: 1
---
# Inference with PyTorch
{: .no_toc }

Learn about PyTorch and how to perform inference with PyTorch models.

PyTorch dominates the deep learning landscape with its readily digestible and flexible API; the large number of ready-made models available, particularly in the natural language (NLP) domain; as well as its domain specific libraries.

With its growing ecosystem of developers and applications, this articles runs through inference with PyTorch, including optimizations that you can make along the way as well as options for deploying your PyTorch model so that you can use it to inference.

This article assumes that you are looking for information about performing inference with your PyTorch model rather than how to train a PyTorch model.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Overview of PyTorch

At the heart of PyTorch is the `nn.Module`, a class that represents an entire deep learning model, or a single layer. Modules can be composed or extended to build models. To write your own module, you implement a forward function that calculates outputs based on inputs, as well as the trained weights of the model. If you are writing your own PyTorch model, then you are likely training it too. Alternatively you can use pre-trained models from PyTorch itself or from other libraries, such as HuggingFace.

To code an image processing model using PyTorch itself:

```bash
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)
```

To create a language model using the HuggingFace library you can:

```bash
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
model = transformers.BertForQuestionAnswering.from_pretrained(model_name)
```

Once you have created or imported a trained model, how do you run it to perform inference? There are a number of different methods and frameworks that you can use to perform inference in PyTorch.

## Inference with native PyTorch

If you are running in an environment that contains Python executables and libraries, you can run your application in native PyTorch.

Once you have your trained model, there are two methods that you (or your data science team) can use to save and load the model for inference:

1. Save and load the entire model

   ```python
   # Save the entire model to PATH
   torch.save(model, PATH)

   # Load the model from PATH and set eval mode for inference
   model = torch.load(PATH)
   model.eval()
   ```

2. Save the parameters of the model, redeclare the model, and load the parameters

   ```python
   # Save the model parameters
   torch.save(model.state_dict(), PATH)

   # Redeclare the model and load the saved parameters
   model = TheModel(...)
   model.load_state_dict(torch.load(PATH))
   model.eval()
   ```

Which of these methods you use depends on your configuration. Saving and loading the entire means that you do not have to redeclare the model, or even have access to the model code itself. But the trade off is that both the saving environment and loading environment have to match in terms of the classes, methods and parameters available (as these are directly serialized and deserialized).

Saving the trained parameters of the model (the state dictionary, or state_dict) is more flexible than the first approach as long as you have access to the original model code.

## Inference with TorchScript

If you are running in an environment that is more constrained and you cannot install PyTorch or other python libraries, you have the option of performing inference with PyTorch models that have been converted to TorchScript. TorchScript is a subset of Python that allows you to create serializable models that can be loaded and executed in non Python environments. It is also optimized for the language constructs most used by deep learning models.

```python
# Export to TorchScript
script = torch.jit.script(model, example)

# Save scripted model
script.save(PATH)
```

```python
# Load scripted model
model = torch.jit.load(PATH)
model.eval()
```

```cpp
#include <torch/script.h>

...

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule
    module = torch::jit::load(PATH);
  }
  catch (const c10::Error& e) {
    ...
  }

...
```

## Inference with ONNXRuntime

When performance is paramount you can use ONNXRuntime to perform inference on a PyTorch model. With ONNXRuntime, you can reduce latency and memory and increase throughput. You can also run a single model on cloud, edge, web  or mobile, using the language bindings and libraries provided with ONNXRuntime.

The first step is to export your PyTorch model to ONNX format using the PyTorch ONNX exporter.

```python
# Specify example data
example = ... 

# Export model to ONNX format
torch.onnx.export(model, PATH, example)
```

Once exported to ONNX format, you can view the model in the Netron viewer to understand the model graph and the inputs and output node names and shapes, and which nodes have variably sized inputs and outputs (dynamic axes).

Then you can run the ONNX model in the environment of your choice. The ONNXRuntime engine is implemented in C++ and had a C++ API as well as APIs in Python, C#, Java, Javascript, Julia and Ruby. For example, the following code snippet shows a skeleton of a C++ inference application.

```cpp
  // Allocate ONNXRuntime session
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Env env;
  Ort::Session session{env, L"model.onnx", Ort::SessionOptions{nullptr}};

  // Allocate model inputs: fill in shape and size
  std::array<float, ...> input{};
  std::array<int64_t, ...> input_shape{...};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
  const char* input_names[] = {...};

  // Allocate model outputs: fill in shape and size
  std::array<float, ...> output{};
  std::array<int64_t, ...> output_shape{...};
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), output_shape.data(), output_shape.size());
  const char* output_names[] = {...};

  // Run the model
  session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
```

Out of the box, ONNXRuntime applies a series of optimizations to the ONNX graph, combining nodes where possible and factoring out constant values (constant folding). ONNXRuntime also integrates with a number of hardware accelerators via its Execution Provider interface, including CUDA, TensorRT, OpenVINO, CoreML and NNAPI, depending on which hardware you are running on.

You can also improve the performance of the ONNX model by quantizing it.

If the application is running in constrained environments, such as mobile and edge you can build a reduced size runtime, based on the model or models that the application runs.


## Further reading

### Convert model to ONNX

* [Basic PyTorch export through torch.onnx](https://pytorch.org/docs/stable/onnx.html)
* [Super-resolution with ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [Export PyTorch model with custom ops](../export-pytorch-model.md)

### PyTorch inference examples

* [Accelerate BERT model on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb)
* [Accelerate BERT model on GPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)
* [Accelerate reduced size BERT model through quantization](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb)

* [Accelerate GPT2 on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb)
* [Accelerate GPT2 (with one step search) on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2-OneStepSearch_OnnxRuntime_CPU.ipynb)
