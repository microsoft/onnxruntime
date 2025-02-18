---
title: Run on Snapdragon devices 
description: Learn how to run Phi-3.5 and Llama 3.2 ONNX models on Snapdragon devices 
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order: 6
---


# Run models on Snapdragon devices with NPUs

Learn how to run SLMs on Snapragon devices with ONNX Runtime.

## Models
Devices with Snapdragon NPUs requires models in a specific size and format.

Models supported currently are:
* [Phi-3.5 mini instruct](https://github.com/microsoft/ort_npu_samples/releases/tag/v73-phi-3.5-2.31)
* [Llama 3.2 3B](https://github.com/microsoft/ort_npu_samples)

Due to Meta licensing restrictions, the Llama model cannot be pre-published. Instructions to generate the Llama model can be found in the link.


## Python application

If your device has Python installed, you can run a simple question and answering script to query the model.

### Install the runtime

```powershell
pip install onnxruntime-genai
```

### Download the script

```powershell
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-qa.py -o model-qa.py
```

### Run the script

```powershell
python .\model-qa.py -e cpu -g -v --system_prompt "You are a helpful assistant. Be brief and concise." --chat_template "<|user|>\n{input} <|end|>\n<|assistant|>" -m ..\..\models\microsoft\phi-3.5-mini-instruct-npu-qnn-2.31-v2
```

## C# Application








