// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "gsl/gsl"
#include "reshape_helper.h"
#include "utils.h"

namespace onnxruntime {

class Reshape final : public OpKernel {
 public:
  explicit Reshape(const OpKernelInfo& info) : OpKernel(info),
                                               allow_zero_(info.GetAttrOrDefault<int64_t>("allowzero", 0) == 1) {
  }

  Status Compute(OpKernelContext* context) const override {
    // Copy the second input tensor into the shape vector
    const auto* shapeTensor = context->Input<Tensor>(1);
    ORT_ENFORCE(shapeTensor->Shape().NumDimensions() == 1,
                "A shape tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(shapeTensor->Shape()[0]);
    const auto* data = shapeTensor->template Data<int64_t>();
    std::vector<int64_t> shape(data, data + nDims);

    const auto* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape, allow_zero_);

    Tensor* Y = context->Output(0, TensorShape(shape));

    CopyCpuTensor(X, Y);

    return Status::OK();
  }

 private:
  const bool allow_zero_;
};

class Reshape_1 final : public OpKernel {
 public:
  explicit Reshape_1(const OpKernelInfo& info) : OpKernel(info) {
    Status status = info.GetAttrs<int64_t>("shape", shape_);
    ORT_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status Compute(OpKernelContext* context) const override {
    std::vector<int64_t> shape = shape_;
    const auto* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();

    ReshapeHelper helper(X_shape, shape);

    Tensor* Y = context->Output(0, TensorShape(shape));

    CopyCpuTensor(X, Y);

    return Status::OK();
  }

 private:
  std::vector<int64_t> shape_;
};

}  //namespace onnxruntime
