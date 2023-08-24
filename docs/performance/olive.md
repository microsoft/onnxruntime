---
title: End to end optimization with Olive
description: Hardware-aware model optimization tool
parent: Performance
nav_order: 5
---
# Olive - hardware-aware model optimization tool

[Olive](https://github.com/microsoft/Olive) is an easy-to-use hardware-aware model optimization tool that composes industry-leading techniques across 
model compression, optimization, and compilation. It works with ONNX Runtime as an E2E inference optimization solution. 

Given a model and targeted hardware, Olive composes the best suitable optimization techniques to output 
the most efficient model(s) and runtime configurations for inferencing with ONNX Runtime, while taking a set of constraints such as accuracy and latency into consideration. Techniques Olive has integrated include ONNX Runtime Transformer optimizations, ONNX Runtime performance tuning, HW-dependent tunable post training quantization, quantize aware training, and more. Olive is the recommended tool for model optimization for ONNX Runtime.

**Examples:**
1. [BERT optimization on CPU (with post training quantization)](https://github.com/microsoft/Olive/blob/main/examples/bert/bert_ptq_cpu.json)
2. [BERT optimization on CPU (with quantization aware training)](https://github.com/microsoft/Olive/blob/main/examples/bert/bert_qat_customized_train_loop_cpu.json)


For more details, pls refer to [Olive repo](https://github.com/microsoft/Olive) and [Olive documentation](https://microsoft.github.io/Olive).