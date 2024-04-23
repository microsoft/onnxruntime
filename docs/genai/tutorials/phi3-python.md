---
title: Python phi-3 tutorial
description: Small but mighty. Run Phi-3 with ONNX Runtime.
has_children: false
parent: Tutorials
grand_parent: Generative AI (Preview)
nav_order: 1
---

# Run the Phi-3 Mini models with the ONNX Runtime generate() API


## Download the model 

Phi-3 ONNX models are published on HuggingFace.

Download either or both of the [short](https://aka.ms/phi3-mini-4k-instruct-onnx) and [long](https://aka.ms/phi3-mini-128k-instruct-onnx) context Phi-3 mini models.


For the short context model.

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
```

For the long context model

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx
```

These model repositories have models that run with DirectML, CPU and CUDA.

## Install the generate() API package

### DirectML

```
pip install numpy
pip install --pre onnxruntime-genai-directml
```

### CPU

```
pip install numpy
pip install --pre onnxruntime-genai
```

### CUDA

```
pip install numpy
pip install --pre onnxruntime-genai-cuda --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Run the model

Run the model with [this script](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

The script accepts a model folder and takes the generation parameters from the config in that model folder. You can also override the parameters on the command line.

This example is using the long context model running with DirectML on Windows.

```bash
python model-qa.py -m models/phi3-mini-128k-instruct-directml-int4-awq-block-128
```

Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

```bash
Input: <|user|>Tell me a joke about creative writing<|end|><|assistant|>
 
Output:  Why don't writers ever get lost? Because they always follow the plot! 
```
