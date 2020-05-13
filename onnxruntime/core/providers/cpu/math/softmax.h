// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl-lite.hpp"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/softmax_shared.h"

namespace onnxruntime {
template <typename T>
class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info}, axis_{1} {
    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    }

    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    auto* Y = ctx->Output(0, X_shape);

    // edge case. one or more dims with value of 0. nothing to do
    if (X_shape.Size() == 0) {
      return Status::OK();
    }

    const int64_t axis = HandleNegativeAxis(axis_, X_shape.NumDimensions());
    const size_t N = X_shape.SizeToDimension(axis);
    const size_t D = X_shape.SizeFromDimension(axis);

    concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

    return SoftmaxCPU(N, D, X->template Data<T>(), Y->template MutableData<T>(), log_softmax_, thread_pool);
  }

 private:
  int axis_;
  bool log_softmax_;
};

}  // namespace onnxruntime
