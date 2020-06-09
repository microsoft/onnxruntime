// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ __inline__ T ComputeGeluGradScalar(T dY, T X) {
  const T kAlpha = T(M_2_SQRTPI) * T(M_SQRT1_2) * T(0.5);
  return dY * (_Normcdf(X) + X * kAlpha * _Exp(-T(0.5) * X * X));
}

template <typename T>
__device__ __inline__ T ComputeGeluApproximationGradScalar(T dY, T X) {
  const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
  const T kGamma = T(0.044715);
  const T kBeta = kAlpha * kGamma * T(3);
  const auto tanh_value =
      static_cast<T>(_Tanh(kAlpha * ((kGamma * X * X * X) + X)));
  return dY * T(0.5) * ((-X * tanh_value * tanh_value + X) * (kBeta * X * X + kAlpha) + T(1) + tanh_value);
}

}  // namespace cuda
}  // namespace onnxruntime
