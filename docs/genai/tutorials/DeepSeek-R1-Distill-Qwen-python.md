---
title: DeepSeek-R1-Distill tutorial
description: Learn how to chat with DeepSeek-R1-Distill ONNX models on your device. 
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order:
---

# Reasoning in Python with DeepSeek-R1-Distill models

## 1. Pre-Requisites: Make a virtual environment and install ONNX Runtime GenAI
```bash
# Installing onnxruntime-genai, olive, and dependencies for CPU
python -m venv .venv && source .venv/bin/activate
pip install requests numpy --pre onnxruntime-genai olive-ai
```

```bash
# Installing onnxruntime-genai, olive, and dependencies for CUDA GPU
python -m venv .venv && source .venv/bin/activate
pip install requests numpy --pre onnxruntime-genai-cuda "olive-ai[gpu]"
```

## 2. Acquire model

Choose your model and convert to ONNX. Note that many LLMs work, so feel free to try with other models too:

```bash
# Using Olive auto-opt to pull a huggingface model, optimize for CPU, and quantize to INT4 using RTN. 
olive auto-opt --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_path ./deepseek-r1-distill-qwen-1.5B --device cpu --provider CPUExecutionProvider --precision int4 --use_model_builder --log_level 1
```

```bash
# Using Olive auto-opt to pull a huggingface model, optimize for CUDA GPUs, and quantize to INT4 using RTN. 
olive auto-opt --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_path ./deepseek-r1-distill-qwen-1.5B --device gpu --provider CUDAExecutionProvider --precision int4 --use_model_builder --log_level 1
```

OR download directly using the Huggingface CLI: 

```bash
# Download the model directly using the huggingface cli
huggingface-cli download onnxruntime/DeepSeek-R1-Distill-ONNX --include 'deepseek-r1-distill-qwen-1.5B/*' --local-dir .
```

## 3. Play with your model on device!
```bash
# CPU Chat inference. If you pulled the model from huggingface, adjust the model directory (-m) accordingly 
curl -o https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-chat.py
python model-chat.py -m deepseek-r1-distill-qwen-1.5B/model -e cpu --chat_template "<|begin▁of▁sentence|><|User|>{input}<|Assistant|>"
```

```bash
# GPU Chat inference. If you pulled the model from huggingface, adjust the model directory (-m) accordingly 
curl -o https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/examples/python/model-chat.py
python model-chat.py -m deepseek-r1-distill-qwen-1.5B/model -e cuda --chat_template "<|begin▁of▁sentence|><|User|>{input}<|Assistant|>"
