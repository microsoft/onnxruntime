---
title: Export PyTorch model
nav_exclude: true
---

## Export PyTorch model to ONNX
{: .no_toc }

This guide covers exporting PyTorch models to ONNX format and running them with ONNX Runtime.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

### Basic Export

Use `torch.onnx.export()` to convert a PyTorch model to ONNX:

```python
import torch
import onnxruntime

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# Run with ONNX Runtime
sess = onnxruntime.InferenceSession("model.onnx")
results = sess.run(None, {"input": dummy_input.numpy()})
print(results)
```

### Handling External Data (Large Models)

When exporting large models with opset 17 or higher, ONNX automatically splits the output into two files if the model exceeds 2 GB:

- `model.onnx` — model graph and metadata
- `model.onnx.data` — raw tensor data (weights, biases, etc.)

**`InferenceSession` will fail if `model.onnx.data` is missing.** Always deploy both files together:

```python
# This will fail if model.onnx.data is absent
sess = onnxruntime.InferenceSession("model.onnx")
# RuntimeError: ... external data file "model.onnx.data" not found
```

To merge the split files into a single, self-contained `.onnx` file:

```python
import onnx

model = onnx.load("model.onnx", load_external_data=False)
onnx.save(
    model,
    "model_combined.onnx",
    save_external_data=False,
    all_tensors_to_one_file=True,
    convert_attribute=True,
)
```

Now `model_combined.onnx` is a single file and can be deployed without the `.data` companion.

### Reducing Model Size

Large exported ONNX models can be optimized to reduce file size:

```python
from onnxruntime.transformers import optimizer

opt = optimizer.optimize_model("model.onnx", model_type="bert", num_heads=12, hidden_size=768)
opt.save_model_to_file("model_optimized.onnx")
```

For more details, see the [ONNX Runtime optimization documentation](../performance/model-optimizations/onnx.md).

---

## Export with custom ONNX operators
{: .no_toc }

This section explains how to export PyTorch models with custom ONNX Runtime ops. The aim is to export a PyTorch model with operators that are not supported in ONNX, and extend ONNX Runtime to support these custom ops.

### Export Built-In Contrib Ops

"Contrib ops" refers to the set of custom ops that are built in to most ORT packages.
Symbolic functions for all contrib ops should be defined in [pytorch_export_contrib_ops.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/pytorch_export_contrib_ops.py).

To export using those contrib ops, call `pytorch_export_contrib_ops.register()` before calling `torch.onnx.export()`. For example:

```python
from onnxruntime.tools import pytorch_export_contrib_ops
import torch

pytorch_export_contrib_ops.register()
torch.onnx.export(...)
```

### Export a Custom Op

To export a custom op that's not a contrib op, or that's not already included in `pytorch_export_contrib_ops`, one will need to
write and register a custom op symbolic function.

We take the Inverse operator as an example:

```python
from torch.onnx import register_custom_op_symbolic

def my_inverse(g, self):
    return g.op("com.microsoft::Inverse", self)

# register_custom_op_symbolic('<namespace>::inverse', my_inverse, <opset_version>)
register_custom_op_symbolic('::inverse', my_inverse, 1)
```

`<namespace>` is a part of the torch operator name. For standard torch operators, namespace can be omitted.

`com.microsoft` should be used as the custom opset domain for ONNX Runtime ops. You can choose the custom opset version during op registration.

For more on writing a symbolic function, see the [torch.onnx documentation](https://pytorch.org/docs/master/onnx.html#adding-support-for-operators).

### Extend ONNX Runtime with Custom Ops

The next step is to add an op schema and kernel implementation in ONNX Runtime.
See [custom operators](../reference/operators/add-custom-op.md) for details.

### Test End-to-End: Export and Run

Once the custom op is registered in the exporter and implemented in ONNX Runtime, you should be able to export it and run it with ONNX Runtime.

Below you can find a sample script for exporting and running the inverse operator as part of a model.

The exported model includes a combination of ONNX standard ops and the custom ops.

This test also compares the output of PyTorch model with ONNX Runtime outputs to test both the operator export and implementation.

```python
import io
import numpy
import onnxruntime
import torch


class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x) + x

x = torch.randn(3, 3)

# Export model to ONNX
f = io.BytesIO()
torch.onnx.export(CustomInverse(), (x,), f)

model = CustomInverse()
pt_outputs = model(x)

# Run the exported model with ONNX Runtime
ort_sess = onnxruntime.InferenceSession(f.getvalue())
ort_inputs = dict((ort_sess.get_inputs()[i].name, input.cpu().numpy()) for i, input in enumerate((x,)))
ort_outputs = ort_sess.run(None, ort_inputs)

# Validate PyTorch and ONNX Runtime results
numpy.testing.assert_allclose(pt_outputs.cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
```

By default, the opset version will be set to `1` for custom opsets. If you'd like to export your
custom op to a higher opset version, you can specify the custom opset domain and version using
the `custom_opsets argument` when calling the export API. Note that this is different than the opset
version associated with default `ONNX` domain.

```python
torch.onnx.export(CustomInverse(), (x,), f, custom_opsets={"com.microsoft": 5})
```

Note that you can export a custom op to any version >= the opset version used at registration.
