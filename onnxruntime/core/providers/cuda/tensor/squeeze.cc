// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Squeeze);

// explicit support for negative axis.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Squeeze);

ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
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
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, count * element_bytes, cudaMemcpyDeviceToDevice));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
