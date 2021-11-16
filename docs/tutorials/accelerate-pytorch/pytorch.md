---
title: Accelerate PyTorch Inference
parent: Accelerate PyTorch
grand_parent: Tutorials
nav_order: 1
---
# Accelerate PyTorch model inferencing
{: .no_toc }

ONNX Runtime can be used to accelerate PyTorch models inferencing.

## Convert model to ONNX
{: .no_toc }

* [Basic PyTorch export through torch.onnx](https://pytorch.org/docs/stable/onnx.html)
* [Super-resolution with ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [Export PyTorch model with custom ops](../export-pytorch-model.md)

## Accelerate PyTorch model inferencing
{: .no_toc }

### BERT
* [Accelerate BERT model on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb)
* [Accelerate BERT model on GPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)
* [Accelerate reduced size BERT model through quantization](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb)

### GPT-2
* [Accelerate GPT2 on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb)
* [Accelerate GPT2 (with one step search) on CPU](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2-OneStepSearch_OnnxRuntime_CPU.ipynb)
