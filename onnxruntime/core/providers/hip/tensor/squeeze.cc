// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

namespace onnxruntime {
namespace hip {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    1, 10,
    kHipExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Squeeze);

// explicit support for negative axis.
ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    11,
    kHipExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Squeeze);

Status Squeeze::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  std::vector<int64_t> output_shape = ComputeOutputShape(X_shape, axes_);

  Tensor* Y = ctx->Output(0, TensorShape(output_shape));

  const void* input = X->DataRaw();
  void* output = Y->MutableDataRaw();
  if (input == output)
    return Status::OK();

  auto count = X->Shape().Size();
  auto element_bytes = X->DataType()->Size();
  HIP_RETURN_IF_ERROR(hipMemcpyAsync(output, input, count * element_bytes, hipMemcpyDeviceToDevice));

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
