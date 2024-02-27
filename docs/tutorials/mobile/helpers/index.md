---
title: ORT Mobile Model Export Helpers
descriptions: Helpers to assist with export and usage of models with ORT Mobile
parent: Deploy on mobile
grand_parent: Tutorials
nav_order: 5
---

# ORT Mobile Model Export Helpers
{: .no_toc }


There are a range of tools available to aid with exporting and analyzing a model for usage with ORT Mobile.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## ORT Mobile model usability checker

The model usability checker provides information on how well a model is likely to run with ORT mobile, including the suitability for using NNAPI on Android and CoreML on iOS. It can also recommend running specific tools to update the model so that it works better with ORT Mobile.

See [here](./model-usability-checker.md) for more details.


## ONNX model opset updater

The ORT Mobile pre-built package only supports the most recent ONNX opsets in order to minimize binary size. Most ONNX models can be updated to a newer ONNX opset using this tool. It is recommended to use the latest opset the pre-built package supports, which is currently opset 15.

The ONNX opsets supported by the pre-built package are documented [here](../../../reference/operators/MobileOps.md).

Usage:

```
python -m onnxruntime.tools.update_onnx_opset --help
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


## ONNX model dynamic shape fixer

If the model can potentially be used with NNAPI or CoreML it may require the input shapes to be made 'fixed' by setting any dynamic dimension sizes to specific values. 

See the documentation on [onnxruntime.tools.make_dynamic_shape_fixed](./make-dynamic-shape-fixed.md) for information on how to do this.


## QDQ format model helpers

Depending on the source of a QDQ format model, it may be necessary to optimize aspects of it to ensure optimal performance with ORT. 
The onnxruntime.tools.qdq_helpers.optimize_qdq_model helper can be used to do this.

Usage:

```
python -m onnxruntime.tools.qdq_helpers.optimize_qdq_model --help
usage: optimize_qdq_model.py [-h] input_model output_model

Update a QDQ format ONNX model to ensure optimal performance when executed using ONNX Runtime.

positional arguments:
  input_model   Provide path to ONNX model to update.
  output_model  Provide path to write updated ONNX model to.

optional arguments:
  -h, --help    show this help message and exit
```

Note that if there are no optimizations the output_model will be the same as the input_model and can be discarded.


## PyTorch export helpers

When exporting a model from [PyTorch](https://pytorch.org/) using [torch.onnx.export](https://pytorch.org/docs/stable/onnx.html) the names of the model inputs can be specified, and the model inputs need to be correctly assembled into a tuple. The infer_input_info helper can be used to automatically discover the input names used in the PyTorch model, and to format the inputs correctly for usage with torch.onnx.export.

In the below example we provide the necessary input to run the torchvision mobilenet_v2 model. 
The input_names and inputs_as_tuple returned can be directly used in the torch.onnx.export call. 
This provides the most benefit when there are multiple inputs to the model, and/or if those inputs involve more complex data types such as dictionaries.


```python
import torch
import torchvision
from onnxruntime import tools

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

input0 = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
input_names, inputs_as_tuple = tools.pytorch_export_helpers.infer_input_info(model, input0)

# input_names and inputs_as_tuple can be directly passed to torch.onnx.export
torch.onnx.export(model, inputs_as_tuple, "model.onnx", input_names=input_names, ...)
```
