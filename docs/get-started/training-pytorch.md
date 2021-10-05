---
title: ORT Training with PyTorch
parent: Get Started
nav_order: 9
---

# Get started with ORT for Training API (PyTorch)
{: .no_toc }
The ORT Training API is a PyTorch frontend that implements the torch.nn.Module interface.


## ORT Training Example
In this example we will go over how to use ORT for Training a model with PyTorch.

```
pip install torch-ort
python -m torch_ort.configure
```

**Note**: This installs the default version of the `torch-ort` and `onnxruntime-training` packages that are mapped to specific versions of the CUDA libraries. Refer to the install options in [ONNXRUNTIME.ai](https://onnxruntime.ai).

 - Add ORTModule in the `train.py`
```python
   from torch_ort import ORTModule
   .
   .
   .
   model = ORTModule(model)
```


## Samples
[ONNX Runtime Training Examples](https://github.com/microsoft/onnxruntime-training-examples)