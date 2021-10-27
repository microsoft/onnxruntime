---
title: Use custom operators
parent: Operators
grand_parent: Reference
nav_order: 3
---
# Custom operators
{: .no_toc }

ONNX Runtime provides options to run custom operators that are not official ONNX operators.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Register a custom operator
A new op can be registered with ONNX Runtime using the Custom Operator API in [onnxruntime_c_api](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h).

1. Create an OrtCustomOpDomain with the domain name used by the custom ops.
2. Create an OrtCustomOp structure for each op and add them to the OrtCustomOpDomain with OrtCustomOpDomain_Add.
3. Call OrtAddCustomOpDomain to add the custom domain of ops to the session options.


## Examples
{: .no_toc}

* [C++ helper API](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc): custom ops `MyCustomOp` and `SliceCustomOp` use the [C++ helper API](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h). The test file also demonstrates an option to  compile the custom ops into a shared library to be used to run a model via the C++ API.

* [Custom op shared library](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/testdata/custom_op_library/custom_op_library.cc): sample custom op shared library containing two custom kernels.

* [Custom op shared library with Python API](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/python/onnxruntime_test_python.py): `testRegisterCustomOpsLibrary` uses the Python API to register a shared library with custom op kernels. Currently, the only supported Execution Providers (EPs) for custom ops registered via this approach are CUDA and CPU.

* [E2E example: Export PyTorch model with custom ONNX operators](../../tutorials/export-pytorch-model.md).

## CUDA custom ops
When a model is run on a GPU, ONNX Runtime will insert a `MemcpyToHost` op before a CPU custom op and append a `MemcpyFromHost` after it to make sure tensors are accessible throughout calling.

When using CUDA custom ops, to ensure synchronization between ORT's CUDA kernels and the custom CUDA kernels, they must all use the same CUDA compute stream. To ensure this, you may first create a CUDA stream and pass it to the underlying Session via SessionOptions (use the `OrtCudaProviderOptions` struct). This will ensure ORT's CUDA kernels use that stream and if the custom CUDA kernels are launched using the same stream, synchronization is now taken care of implicitly.

For example, see how the afore-mentioned `MyCustomOp` is being launched and how the Session using this custom op is created.


## Contrib ops

The [contrib ops domain](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/contrib_ops) contains ops that are built in to the runtime by default. However most new operators should not be added here to avoid increasing binary size of the core runtime package.

See for example the Inverse op added in [#3485](https://github.com/microsoft/onnxruntime/pull/3485).

The custom op's schema and shape inference function should be added in [contrib_defs.cc](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/graph/contrib_ops/contrib_defs.cc) using `ONNX_CONTRIB_OPERATOR_SCHEMA`.

```c++
ONNX_CONTRIB_OPERATOR_SCHEMA(Inverse)
    .SetDomain(kMSDomain) // kMSDomain = "com.microsoft"
    .SinceVersion(1) // Same version used at op (symbolic) registration
    ...
```

A new operator should have complete reference implementation tests and shape inference tests.

Reference implementation python tests should be added in
[onnxruntime/test/python/contrib_ops](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/python/contrib_ops).
E.g., [onnx_test_trilu.py](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/python/contrib_ops/onnx_test_trilu.py)

Shape inference C++ tests should be added in
[onnxruntime/test/contrib_ops](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops).
E.g., [trilu_shape_inference_test.cc](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops/trilu_shape_inference_test.cc)

The operator kernel should be implemented using `Compute` function
under contrib namespace in [onnxruntime/contrib_ops/cpu/](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cpu/)
for CPU and [onnxruntime/contrib_ops/cuda/](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cuda/) for CUDA.

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

The kernel should be registered in [cpu_contrib_kernels.cc](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cpu_contrib_kernels.cc) for CPU and [cuda_contrib_kernels.cc](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/contrib_ops/cuda_contrib_kernels.cc) for CUDA.

Now you should be able to build and install ONNX Runtime to start using your custom op.

### Contrib Op Tests

Tests should be added in [onnxruntime/test/contrib_ops/](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/contrib_ops/).
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
