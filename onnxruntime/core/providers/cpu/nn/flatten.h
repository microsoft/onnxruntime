// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "gsl/gsl_util"
#include "core/providers/cpu/tensor/utils.h"

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
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.NumDimensions()) >= axis_, "The rank of input tensor must be >= axis");

    Tensor* Y = context->Output(0, TensorShape({X_shape.SizeToDimension(axis_), X_shape.SizeFromDimension(axis_)}));

    CopyCpuTensor(X, Y);

    return Status::OK();
  }

 private:
  int64_t axis_;
};
}  // namespace onnxruntime
