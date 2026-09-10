---
title: Export PyTorch model
nav_exclude: true
---

## Export PyTorch model with custom ONNX operators
{: .no_toc }

This document explains the process of exporting PyTorch models with custom ONNX Runtime ops. The aim is to export a PyTorch model with operators that are not supported in ONNX, and extend ONNX Runtime to support these custom ops.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

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

## Handling External Data Files

When exporting PyTorch models using `torch.onnx.export` with opset 17+,
PyTorch may automatically split the export into two files:

- `model.onnx` — the model graph structure
- `model.onnx.data` — the weight tensors stored as external data

Both files must be present in the same directory for ONNX Runtime to load
the model correctly. `InferenceSession` will fail if `model.onnx.data` is missing.

```python
# Fails if model.onnx.data is not in the same directory
session = onnxruntime.InferenceSession("model.onnx")
```

### Option A — Models under 2 GB: merge into a single file

For models whose weights fit within protobuf's 2 GB limit, the split files can be
merged into a single self-contained `.onnx` file. This simplifies distribution
to Docker images, cloud buckets, and API endpoints:

```python
import onnx

model = onnx.load("model.onnx", load_external_data=True)
onnx.save(model, "model_single.onnx", save_as_external_data=False)
```

> **Note:** Requires the `onnx` package (`pip install onnx`). Both `model.onnx`
> and `model.onnx.data` must be in your working directory before running this.

### Option B — Models 2 GB and above: deploy as a file pair

When model weights exceed protobuf's 2 GB limit, merging is not possible.
The `.onnx.data` file is the ONNX spec's mechanism for handling this case.
Both files must be co-located at inference time:

```python
# Both model.onnx and model.onnx.data must be in the same directory
session = onnxruntime.InferenceSession("/path/to/model/model.onnx")
# ONNX Runtime resolves model.onnx.data relative to the .onnx file location
```

For cloud storage (e.g. GCS, S3), upload both files to the same prefix and
download both to the same local directory before loading:

```bash
gsutil cp model.onnx model.onnx.data gs://your-bucket/models/
```
