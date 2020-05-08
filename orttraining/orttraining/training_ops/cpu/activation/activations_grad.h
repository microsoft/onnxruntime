// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

#include <unsupported/Eigen/SpecialFunctions>

namespace onnxruntime {
namespace contrib {

#ifndef EIGEN_X
#define EIGEN_X ConstEigenVectorArrayMap<T>(X->template Data<T>(), X->Shape().Size())
#endif

#ifndef EIGEN_X_VAR
#define EIGEN_X_VAR(var) ConstEigenVectorArrayMap<T> var(X->template Data<T>(), X->Shape().Size())
#endif

#ifndef EIGEN_Y
#define EIGEN_Y EigenVectorArrayMap<T>(Y->template MutableData<T>(), Y->Shape().Size())
#endif

#ifndef EIGEN_Y_VAR
#define EIGEN_Y_VAR(var) EigenVectorArrayMap<T> var(Y->template MutableData<T>(), Y->Shape().Size())
#endif

#ifndef EIGEN_DY_VAR
#define EIGEN_DY_VAR(var) ConstEigenVectorArrayMap<T> var(dY->template Data<T>(), dY->Shape().Size())
#endif

#ifndef EIGEN_DX
#define EIGEN_DX EigenVectorArrayMap<T>(dX->template MutableData<T>(), dX->Shape().Size())
#endif

template <typename T>
class GeluGrad final : public OpKernel {
 public:
  GeluGrad(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* dY = context->Input<Tensor>(0);
    const auto* X = context->Input<Tensor>(1);
    Tensor* dX = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_DY_VAR(dy);
    constexpr T kAlpha = static_cast<float>(M_2_SQRTPI) * static_cast<float>(M_SQRT1_2) * 0.5f;
    EIGEN_DX = dy * (0.5f * ((xm * static_cast<float>(M_SQRT1_2)).erf() + 1.0f) +
                    xm * kAlpha * (-0.5f * xm * xm).exp());

    return Status::OK();
  }
};

template <typename T>
class FastGeluGrad final : public OpKernel {
 public:
  FastGeluGrad(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* dY = context->Input<Tensor>(0);
    const auto* X = context->Input<Tensor>(1);
    Tensor* dX = context->Output(0, X->Shape());
    //
    // Commented out EIGEN implentation due to EIGEN bug.
    // On Windows Release build with GPU enabled, kAlpha * EIGEN_X below would produce pure 0
    // result, even though neither kAlpha nor EIGEN_X is zero.
    // Given that CPU kernel is mostly for conformance check, where performance is not of high
    // priority, to workaround this bug, use a for loop and avoid using EIGEN library.
    //
    // EIGEN_X_VAR(xm);
    // EIGEN_DY_VAR(dy);

    // const auto x_cube = EIGEN_X.cube();
    // const auto tanh_result = ((T)kAlpha * (EIGEN_X + kGamma * x_cube)).tanh();
    // const auto sech_sqr_result = 1 - (tanh_result * tanh_result);

    // EIGEN_DX = dy * (0.5f * (tanh_result + sech_sqr_result * (kAlpha * xm + kBeta * x_cube) + 1));
    //
    const T* dY_data = dY->template Data<T>();
    const T* X_data = X->template Data<T>();
    T* dX_data = dX->template MutableData<T>();
    int64_t elem_count = X->Shape().Size();
    for (auto i = 0; i < elem_count; ++i) {
      const auto x_val = X_data[i];
      const auto x_cube = x_val * x_val * x_val;
      T tanh_result = std::tanh(kAlpha * x_val + kAlpha * kGamma * x_cube);
      T sech_sqr_result = 1 - (tanh_result * tanh_result);
      dX_data[i] = (dY_data[i]) * (0.5f * (tanh_result + sech_sqr_result * (kAlpha * x_val + kBeta * x_cube) + 1));
    }
    return Status::OK();
  }

 private:
  static constexpr T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
  static constexpr T kGamma = 0.044715f;
  static constexpr T kBeta = kGamma * kAlpha * 3.0f;

};

}  // namespace contrib
}  // namespace onnxruntime
