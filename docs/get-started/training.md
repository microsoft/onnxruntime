---
title: Training
parent: Get Started
nav_order: 3
---

# Use ONNX Runtime for Training

The ONNX Runtime training feature enables easy integration with existing Pytorch trainer code to accelerate the execution. With a few lines of code, you can add ONNX Runtime into your existing training scripts and start seeing acceleration. The current preview version supports training acceleration for transformer models on NVIDIA GPUs.

**[ONNX Runtime pre-training sample](https://github.com/microsoft/onnxruntime-training-examples)**: This sample is setup to pre-train the BERT-Large model to show how ONNX Runtime training can be used to accelerate training execution.

## Train PyTorch model with ONNX Runtime

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

## Build ONNX Runtime Training from source

To use ONNX Runtime training in a custom environment, like on-prem NVIDIA DGX-2 clusters, you can use these [build instructions](../how-to/build.md#training) to generate the Python package to integrate into existing trainer code.
