---
title: Generate API (Preview)
description: Run generative models with the ONNX Runtime generate() API
has_children: true
nav_order: 6
---

# ONNX Runtime generate() API

_Note: this API is in preview and is subject to change._

Run generative AI models with ONNX Runtime.

See the source code here: [https://github.com/microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) 

This library provides the generative AI loop for ONNX models, including tokenization and other pre-processing, inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or run each iteration of the model in a loop, generating one token at a time, and optionally updating generation parameters inside the loop.

It has support for greedy/beam search and TopP, TopK sampling to generate token sequences and built-in logits processing like repetition penalties. You can also easily add custom scoring.

Other supported features include applying chat templates and structured output (for tool calling)

