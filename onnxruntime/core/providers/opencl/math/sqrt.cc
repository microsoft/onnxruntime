// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace {

#define CONTENT_NAME sqrt_kernel_src
#include "opencl_generated/math/kernels/sqrt.cl.inc"

}  // namespace

namespace onnxruntime {
namespace opencl {

template <typename T>
class Sqrt : public OpenCLKernel {
 public:
  explicit Sqrt(const OpKernelInfo& info) : OpenCLKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class Sqrt<float> : public OpenCLKernel {
 public:
  explicit Sqrt(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(sqrt_kernel_src, sqrt_kernel_src_len);
    LoadKernel("Sqrt_Float");
  }

  Status Compute(OpKernelContext* context) const override;
};

Status Sqrt<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());
  const int64_t input_size = X->Shape().Size();
  if (input_size == 0)
    return Status::OK();
  ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("Sqrt_Float")}
          .SetBuffers(*X, *Y)
          .Launch(*exec_, {X->SizeInBytes() / 4, 1, 1}));

  return Status::OK();
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Sqrt,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sqrt<float>);

}  // namespace opencl
}  // namespace onnxruntime
