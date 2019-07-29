// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class Clip final : public OpKernel {
 public:
  Clip(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<T>("max", &max_).IsOK());
    ORT_ENFORCE(info.GetAttr<T>("min", &min_).IsOK());
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    EigenVectorMap<T>(Y->template MutableData<T>(), Y->Shape().Size()) =
        ConstEigenVectorMap<T>(X->template Data<T>(), X->Shape().Size())
            .cwiseMax(min_)
            .cwiseMin(max_);
    return Status::OK();
  }

 private:
  T max_;
  T min_;
};

}  // namespace onnxruntime
