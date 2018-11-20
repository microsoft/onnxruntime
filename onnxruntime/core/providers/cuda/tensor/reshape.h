// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "gsl/gsl_util"
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
    ONNXRUNTIME_ENFORCE(shapeTensor->Shape().NumDimensions() == 1,
                "A shape tensor must be a vector tensor.");
    size_t nDims = static_cast<size_t>(shapeTensor->Shape()[0]);
    const int64_t* data = shapeTensor->template Data<int64_t>();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < nDims; ++i)
      shape.push_back(data[i]);

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
};

class Reshape_1 final : public CudaKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : CudaKernel(info) {
    Status status = info.GetAttrs<int64_t>("shape", shape_);
    ONNXRUNTIME_ENFORCE(status.IsOK(), "Attribute shape is not set.");
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
