---
title: ORT Mobile Model Export Helpers
descriptions: 
parent: Mobile
nav_order: 1
---
# ORT Mobile Model Export Helpers
{: .no_toc }


There are a range of tools available to aid with exporting and analyzing a model for usage with ORT Mobile.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

### ORT Mobile model usability checker

The model usability checker provides information on how well a model is likely to run with ORT mobile, including the suitability for using NNAPI on Android and CoreML on iOS. It can also recommend running specific tools to update the model so that it works better with ORT Mobile.

See [here](./) for details

### ONNX model opset updater

The ORT Mobile pre-built package only supports the most recent ONNX opsets in order to minimize binary size. Most ONNX models can be updated to a newer opset. Use this tool to do so. It is recommended to use the latest opset the pre-built package supports, which is currently opset 15.

The opsets supported by the pre-built package are documented [here](../../../reference/operators/mobile_package_op_type_support_1.10.md).

```
python -m onnxruntime.tools.update_onnx_opset -h
usage: update_onnx_opset.py:update_onnx_opset_helper [-h] [--opset OPSET] input_model output_model

Update the ONNX opset of the model. New opset must be later than the existing one. If not specified will update to opset 15.

positional arguments:
  input_model    Provide path to ONNX model to update.
  output_model   Provide path to write updated ONNX model to.

optional arguments:
  -h, --help     show this help message and exit
  --opset OPSET  ONNX opset to update to.
```

Example usage:
```
python -m onnxruntime.tools.update_onnx_opset --opset 15 model.onnx model.opset15.onnx
```

### ONNX model dynamic shape fixer

If the model can potentially be used with NNAPI or CoreML it may require the input shapes to be made 'fixed'. 

See documentation on [onnxruntime.tools.make_dynamic_shape_fixed](./make-dynamic-shape-fixed.md) for information on how to do this.


### PyTorch export helpers

When exporting a model from [PyTorch](https://pytorch.org/) using [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html) the names of the graph inputs can be specified, and the inputs needs to be assembled into a tuple. The infer_input_info helper can be used to automatically discover the input names, and to format the inputs correctly for usage with torch.onnx.export.

```python
import torch
import torchvision
from onnxruntime import tools

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

input0 = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
input_names, inputs_as_tuple = tools.pytorch_export_helpers.infer_input_info(model, input0)
torch.onnx.export(model, inputs_as_tuple, "model.onnx", input_names=input_names, ...)
```
