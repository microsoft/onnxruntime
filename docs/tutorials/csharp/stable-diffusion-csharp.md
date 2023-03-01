---
title: Stable Diffusion with C# 
description: We will learn how to use Stable Diffusion in a C# Console App.
parent: Stable Diffusion with C#
grand_parent: Tutorials
has_children: false
nav_order: 1
---


# Inference with C# Stable Diffusion and ONNX Runtime
{: .no_toc }

In this tutorial we will learn how to do inferencing for the popular Stable Diffusion deep learning model in C#.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisites
This tutorial can be run locally or by leveraging Azure Machine Learning compute.

- [Download the Source Code](https://github.com/cassiebreviu/StableDiffusion)

To run locally:

- [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- [VS Code](https://code.visualstudio.com/Download) with the [Jupyter notebook extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
- [Anacaonda](https://www.anaconda.com/)
- A GPU enabled machine with CUDA EP Configured. This was built on a GTX 3070 and it has not been tested on anything smaller. Follow this tutorial to configure CUDA and cuDNN for GPU with ONNX Runtime and C# on Windows 11 [here](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)

To run in the cloud with Azure Machine Learning:

- [Azure Subscription](https://azure.microsoft.com/free/)
- [Azure Machine Learning Resource](https://azure.microsoft.com/services/machine-learning/)

## Use Hugging Face to download the Stable Diffusion models version of choice

Hugging Face has a great library of open source model, for Stable Diffusion we will be downloading the [ONNX Stable Diffusion models from Hugging Face](https://huggingface.co/models?sort=downloads&search=Stable+Diffusion).

Once you have selected a model, click `Files and Versions`, then select the `ONNX` branch. If there isn't a ONNX model branch avaialble, you can use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

- Clone the repo:
```text
git lfs install
git clone https://huggingface.co/<contributor>/<model-name>
```

- Copy the folders with the ONNX files to the C# project folder. The folders should include: `unet`, `vae_decoder`, `text_encoder`, `safety_checker`.

## Understanding the model in Python

When taking a prebuilt model and operationalizing it, its useful to take a moment and understand the models in this pipeline. This code is based on the Hugging Face Diffusers Library and Blog. We will be inferencing our model with C# but if you want to learn more about how it works [check out this blog post](https://huggingface.co/blog/stable-diffusion).

- The code to test out the model is provided [in this tutorial](https://onnxruntime.ai/docs/tutorials/azureml.html). Check out the source for testing and inferencing this model in Python. Below is a sample `input` sentence and a sample `output` from running the model.

## Inference with C#

### Tokenizization and Embedding
```csharp
 
```
### Create the Tensors

### Create the `input` of `List<NamedOnnxValue>` that is needed for inference

### Run Inference
- Create the `InferenceSession`, run the inference and print out the result.

```csharp
  // Create an InferenceSession from the Model Path.
  var session = new InferenceSession(modelPath);

  // Run session and send the input data in to get inference output. 
  var output = session.Run(input);
```
### Postprocess the `output` and print the result

## Resources

- [C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
