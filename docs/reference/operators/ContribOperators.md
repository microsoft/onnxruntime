---
title: Contrib operators
parent: Operators
grand_parent: Reference
nav_order: 3
---

# Contrib ops

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

The [contrib ops domain](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops) contains ops that are built in to the runtime by default. Only selected operators are added as contrib ops to avoid increasing the binary size of the core runtime package. When possible, [custom operators](./add-custom-op.md) should be used.

## Contrib Op List

The contrib operator schemas are documented in the ONNX Runtime repository.

| Release | Documentation |
|---------|---------------|
| Main | [https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md) |
| 1.17 | [https://github.com/microsoft/onnxruntime/blob/rel-1.17.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.17.0/docs/ContribOperators.md)|
| 1.16 | [https://github.com/microsoft/onnxruntime/blob/rel-1.16.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.16.0/docs/ContribOperators.md)|
| 1.15 | [https://github.com/microsoft/onnxruntime/blob/rel-1.15.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.15.0/docs/ContribOperators.md)|
| 1.14 | [https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ContribOperators.md)|
| 1.13 | [https://github.com/microsoft/onnxruntime/blob/rel-1.13.1/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.13.1/docs/ContribOperators.md)|
| 1.12 | [https://github.com/microsoft/onnxruntime/blob/rel-1.12.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.12.0/docs/ContribOperators.md)|
| 1.11 | [https://github.com/microsoft/onnxruntime/blob/rel-1.11.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.11.0/docs/ContribOperators.md)|
| 1.10 | [https://github.com/microsoft/onnxruntime/blob/rel-1.10.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.10.0/docs/ContribOperators.md)|
| 1.9 | [https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.9.0/docs/ContribOperators.md)|
| 1.8 | [https://github.com/microsoft/onnxruntime/blob/rel-1.8.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.8.0/docs/ContribOperators.md)
| 1.7 | [https://github.com/microsoft/onnxruntime/blob/rel-1.7.0/docs/ContribOperators.md](https://github.com/microsoft/onnxruntime/blob/rel-1.7.0/docs/ContribOperators.md)|

## Adding Contrib ops

The custom op's schema and shape inference function should be added in [contrib_defs.cc](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/graph/contrib_ops/contrib_defs.cc) using `ONNX_CONTRIB_OPERATOR_SCHEMA`. Example: [Inverse op](https://github.com/microsoft/onnxruntime/pull/3485)

```c++
ONNX_CONTRIB_OPERATOR_SCHEMA(Inverse)
    .SetDomain(kMSDomain) // kMSDomain = "com.microsoft"
    .SinceVersion(1) // Same version used at op (symbolic) registration
    ...
```

A new operator should have complete reference implementation tests and shape inference tests.

Reference implementation python tests should be added in
[onnxruntime/test/python/contrib_ops](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/python/contrib_ops).
E.g., [aten_op_tests.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/python/contrib_ops/aten_op_tests.py)

Shape inference C++ tests should be added in
[onnxruntime/test/contrib_ops](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/contrib_ops).
E.g., [trilu_shape_inference_test.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/cpu/tensor/trilu_shape_inference_test.cc)

The operator kernel should be implemented using `Compute` function
under contrib namespace in [onnxruntime/contrib_ops/cpu/](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/contrib_ops/cpu/)
for CPU and [onnxruntime/contrib_ops/cuda/](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/contrib_ops/cuda/) for CUDA.

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

The kernel should be registered in [cpu_contrib_kernels.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc) for CPU and [cuda_contrib_kernels.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc) for CUDA.

Now you should be able to build and install ONNX Runtime to start using your custom op.

### Contrib Op Tests

Tests should be added in [onnxruntime/test/contrib_ops/](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/contrib_ops/).
For example:

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


