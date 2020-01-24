// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class GeluGrad final : public OpKernel {
 public:
  GeluGrad(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* dY = context->Input<Tensor>(0);
    const auto* X = context->Input<Tensor>(1);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_DY_VAR(dy);
    constexpr T kAlpha = static_cast<float>(M_2_SQRTPI) * static_cast<float>(M_SQRT1_2) * 0.5f;
    EIGEN_Y = dy * (0.5f * ((xm * static_cast<float>(M_SQRT1_2)).erf() + 1.0f) +
                    xm * kAlpha * (-0.5f * xm * xm).exp());

    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
