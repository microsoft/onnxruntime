---
title: Generative AI
description: Run generative models with ONNX Runtime GenAI
has_children: true
nav_order: 6
---

# Generative AI with ONNX Runtime

Run generative AI models with ONNX Runtime.

This library provides the generative AI loop for ONNX models, including inference with ONNX Runtime, logits processing, search and sampling, and KV cache management.

Users can call a high level `generate()` method, or run each iteration of the model in a loop, generating one token at a time, and optionally updating generation parameters inside the loop.

It has support for greedy/beam search and TopP, TopK sampling to generate token sequences and built-in logits processing like repetition penalties. You can also easily add custom scoring.
