---
title: Export PyTorch model
nav_exclude: true 
---

## Export PyTorch model with custom ONNX operators
{: .no_toc }

This document explains the process of exporting PyTorch models with custom ONNX Runtime ops. The aim is to export a PyTorch model with operators that are not supported in ONNX, and extend ONNX Runtime to support these custom ops.

Currently, a torch op can be exported as a custom operator using our custom op (symbolic) registration API. We can  use this API to register custom ONNX Runtime ops under "com.microsoft" domain.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

### Export a Custom Op

In this example, we take Inverse operator as an example. To enable export of ```torch.inverse```, a symbolic function can be created and registered as part of custom ops:

```python
from torch.onnx import register_custom_op_symbolic

def my_inverse(g, self):
    return g.op("com.microsoft::Inverse", self)

# register_custom_op_symbolic('<namespace>::inverse', my_inverse, <opset_version>)
register_custom_op_symbolic('::inverse', my_inverse, 1)
```

`<namespace>` is a part of the torch operator name. For standard torch operators, namespace can be omitted.

`com.microsoft` should be used as the custom opset domain for ONNX Runtime ops. You can choose the custom opset version during op registration.

All symbolics for ONNX Runtime custom ops are defined in `tools/python/register_custom_ops_pytorch_exporter.py`.

If you are adding a symbolic function for a new custom op, add the function to this file.

### Extend ONNX Runtime with Custom Ops

The next step is to add op schema and kernel implementation in ONNX Runtime.
Consider the Inverse custom op as an example added in:
https://github.com/microsoft/onnxruntime/pull/3485

Custom op schema and shape inference function should be added in https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/graph/contrib_ops/contrib_defs.cc using `ONNX_CONTRIB_OPERATOR_SCHEMA`.

```c++
ONNX_CONTRIB_OPERATOR_SCHEMA(Inverse)
    .SetDomain(kMSDomain) // kMSDomain = "com.microsoft"
    .SinceVersion(1) // Same version used at op (symbolic) registration
    ...
```

To comply with ONNX guideline for new operators, a new operator should have complete reference implementation tests and shape inference tests.

Reference implementation python tests should be added in:
https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/python/contrib_ops
E.g.: https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/python/contrib_ops/onnx_test_trilu.py

Shape inference C++ tests should be added in:
https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops
E.g.: https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops/trilu_shape_inference_test.cc

The operator kernel should be implemented using ```Compute``` function
under contrib namespace in `https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cpu/<operator>.cc` 
for CPU and `https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cuda/<operator>.cc` for CUDA.

```c++
namespace onnxruntime {
namespace contrib {

class Inverse final : public OpKernel {
 public:
  explicit Inverse(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override;

 private:
 ...
};

ONNX_OPERATOR_KERNEL_EX(
    Inverse,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16>()),
    Inverse);

Status Inverse::Compute(OpKernelContext* ctx) const {
... // kernel implementation
}

}  // namespace contrib
}  // namespace onnxruntime
```

Operator kernel should be registered in https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cpu_contrib_kernels.cc for CPU and https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cuda_contrib_kernels.cc for CUDA.

Now you should be able to build and install ONNX Runtime to start using your custom op.

### ONNX Runtime Tests

ONNX Runtime custom op kernel tests should be added in: https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops/<operator>_test.cc

```c++
namespace onnxruntime {
namespace test {

// Add a comprehensive set of unit tests for custom op kernel implementation

TEST(InverseContribOpTest, two_by_two_float) {
  OpTester test("Inverse", 1, kMSDomain); // custom opset version and domain
  test.AddInput<float>("X", {2, 2}, {4, 7, 2, 6});
  test.AddOutput<float>("Y", {2, 2}, {0.6f, -0.7f, -0.2f, 0.4f});
  test.Run();
}

...

}  // namespace test
}  // namespace onnxruntime
```

### Test model Export End to End

Once the custom op is registered in the exporter and implemented in ONNX Runtime, you should be able to export it as part of you ONNX model and run it with ONNX Runtime.

Below you can find a sample script for exporting and running the inverse operator as part of a model.

The exported model includes a combination of ONNX standard ops and the custom ops.

This test also compares the output of PyTorch model with ONNX Runtime outputs to test both the operator export and implementation.

```python
import torch
import onnxruntime
import io
import numpy


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

By default, the opset version will be set to ``1`` for custom opsets. If you'd like to export your
custom op to a higher opset version, you can specify the custom opset domain and version using 
the ``custom_opsets argument`` when calling the export API. Note that this is different than the opset 
version associated with default ```ONNX``` domain.

```python
torch.onnx.export(CustomInverse(), (x,), f, custom_opsets={"com.microsoft": 5})
```

Note that you can export a custom op to any version >= the opset version used at registration.

We have a set of tests for export and output validation of ONNX models with ONNX Runtime custom ops in 
``tools/test/test_test_custom_ops_pytorch_exporter.py``. If you're adding a new custom operator, please
make sure to include tests in this file.

You can run these tests using the command:

```bash
PYTHONPATH=<path_to_onnxruntime/tools> pytest -v test_custom_ops_pytorch_exporter.py
```
