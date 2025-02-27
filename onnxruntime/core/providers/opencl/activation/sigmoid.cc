

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sigmoid.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME sigmoid_kernel_src
#include "opencl_generated/activation/kernels/sigmoid.cl.inc"
}  // namespace

template <typename T>
class Sigmoid : public OpenCLKernel {
 public:
  explicit Sigmoid(const OpKernelInfo& info) : OpenCLKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class Sigmoid<float> : public OpenCLKernel {
 public:
  explicit Sigmoid(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(sigmoid_kernel_src, sigmoid_kernel_src_len);
    LoadKernel("Sigmoid_Float");
  }

  Status Compute(OpKernelContext* context) const override;
};

Status Sigmoid<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());
  const int64_t input_size = X->Shape().Size();
  if (input_size == 0)
    return Status::OK();
  ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Sigmoid_Float")}
          .SetBuffers(*X, *Y)
          .Launch(*exec_, {X->SizeInBytes() / 4, 1, 1}));

  return Status::OK();
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Sigmoid,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sigmoid<float>);

}  // namespace opencl
}  // namespace onnxruntime
