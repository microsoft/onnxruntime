// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "gsl/gsl"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {

class Flatten final : public OpKernel {
 public:
  Flatten(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    const TensorShape& X_shape = X->Shape();
    auto axis = axis_;

    // Valid axis range is [-rank, rank] instead of [-rank, rank-1], add additional check to only handle neg axis case.
    if (axis < 0) {
      axis = HandleNegativeAxis(axis, X_shape.NumDimensions());  // handle negative and enforce axis is valid
    }

    ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis, "The rank of input tensor must be >= axis");

    Tensor* Y = context->Output(0, {X_shape.SizeToDimension(axis), X_shape.SizeFromDimension(axis)});

    CopyCpuTensor(X, Y);

    return Status::OK();
  }

 private:
  int64_t axis_;
};
}  // namespace onnxruntime
