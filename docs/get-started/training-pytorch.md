---
title: Large Model Training
parent: Get Started
nav_order: 12
---

# Get started with Large Model Training with ORTModule
{: .no_toc }

`ONNX Runtime Training`'s `ORTModule` offers a high performance training engine for models defined using the `PyTorch` frontend. `ORTModule` is designed to accelerate the training of large models without needing to change the model definition and with just a single line of code change (the `ORTModule` wrap) to the entire training script.

Using the ORTModule class wrapper, ONNX Runtime runs the forward and backward pass of the training script using an optimized automatically-exported ONNX computation graph.

> **Note**: The `torch-ort` package is no longer the recommended installation path. Use `onnxruntime-training` directly as described below.

## Install ONNX Runtime Training

```sh
pip install onnxruntime-training
```

For GPU support:

```sh
pip install onnxruntime-training-gpu
```

## ORT Training Example
In this example we will go over how to use ORT for Training a model with PyTorch.

 - Add ORTModule in the `train.py`

```diff
+  from onnxruntime.training.ortmodule import ORTModule
   .
   .
   .
-  model = build_model() # Users PyTorch model
+  model = ORTModule(build_model())
```

## Samples
[ONNX Runtime Training Examples](https://github.com/microsoft/onnxruntime-training-examples)
