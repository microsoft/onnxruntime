// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatten.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    1, 8,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    9, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

// explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

ONNX_OPERATOR_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

Status Flatten::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  auto axis = axis_;
  // Valid axis range is [-rank, rank] instead of [-rank, rank-1], add additional check to only handle neg axis case.
  if (axis < 0) {
    axis = HandleNegativeAxis(axis, X_shape.NumDimensions());  // handle negative and enforce axis is valid
  }

  ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

  Tensor* Y = ctx->Output(0, {X_shape.SizeToDimension(axis), X_shape.SizeFromDimension(axis)});
  //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();
  if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, X_shape.Size() * X->DataType()->Size(),
                                         cudaMemcpyDeviceToDevice, Stream()));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
