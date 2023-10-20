// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace cuda {

TensorShape InferReshapeOutputShape(
  const Tensor* src,
  const Tensor* shape,
  bool allow_zero
);

Status FuncReshape(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* X,
    const Tensor* shape,
    const bool /*allow_zero*/,
    Tensor* Y);

std::unique_ptr<Tensor> FuncReshape(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* X,
    const Tensor* shape,
    const bool allow_zero
);

class Reshape final : public CudaKernel {
 public:
  Reshape(const OpKernelInfo& info) : CudaKernel(info),
                                      allow_zero_(info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0)) == 1) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    // Copy the second input tensor into the shape vector
    const Tensor* data_tensor = context->Input<Tensor>(0);
    const Tensor* shape_tensor = context->Input<Tensor>(1);
    const auto target_shape = InferReshapeOutputShape(data_tensor, shape_tensor, allow_zero_);
    Tensor* output_tensor = context->Output(0, target_shape);
    return FuncReshape(this, context, data_tensor, shape_tensor, allow_zero_, output_tensor);
  }

 private:
  bool allow_zero_;
};

class Reshape_1 final : public CudaKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : CudaKernel(info) {
    Status status = info.GetAttrs("shape", shape_);
    ORT_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    TensorShapeVector shape = shape_;
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape);

    Tensor* Y = context->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      ORT_ENFORCE(context->GetComputeStream());
      ORT_RETURN_IF_ERROR(CopyTensor(*X, *Y, *context->GetComputeStream()));
    }

    return Status::OK();
  }

 private:
  TensorShapeVector shape_;
};

}  // namespace cuda
}  // namespace onnxruntime
