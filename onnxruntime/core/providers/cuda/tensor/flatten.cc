// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatten.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Flatten);

Status Flatten::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  ONNXRUNTIME_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis_, "The rank of input tensor must be >= axis");

  Tensor* Y = ctx->Output(0, TensorShape({X_shape.SizeToDimension(axis_), X_shape.SizeFromDimension(axis_)}));
  //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  const void* source = X->DataRaw();
  void* target = Y->MutableDataRaw();
  if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, X_shape.Size() * X->DataType()->Size(), cudaMemcpyDeviceToDevice));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
