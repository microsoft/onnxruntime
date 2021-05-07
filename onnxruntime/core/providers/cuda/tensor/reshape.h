// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "gsl/gsl"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace cuda {

class Reshape final : public CudaKernel {
 public:
  Reshape(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    // Copy the second input tensor into the shape vector
    const Tensor* shapeTensor = context->Input<Tensor>(1);
    if (shapeTensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    if (shapeTensor->Shape().NumDimensions() != 1) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "A shape tensor must be a vector tensor, got ", shapeTensor->Shape().NumDimensions(), " dimensions");
    size_t nDims = static_cast<size_t>(shapeTensor->Shape()[0]);
    const int64_t* data = shapeTensor->template Data<int64_t>();
    std::vector<int64_t> shape(data, data + nDims);
    const Tensor* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape);

    Tensor* Y = context->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      CopyTensor(*X, *Y);
    }

    return Status::OK();
  }
};

class Reshape_1 final : public CudaKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : CudaKernel(info) {
    Status status = info.GetAttrs<int64_t>("shape", shape_);
    ORT_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    std::vector<int64_t> shape = shape_;
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape);

    Tensor* Y = context->Output(0, TensorShape(shape));
    const void* source = X->DataRaw();
    void* target = Y->MutableDataRaw();
    //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      CopyTensor(*X, *Y);
    }

    return Status::OK();
  }

 private:
  std::vector<int64_t> shape_;
};

}  // namespace cuda
}  // namespace onnxruntime
