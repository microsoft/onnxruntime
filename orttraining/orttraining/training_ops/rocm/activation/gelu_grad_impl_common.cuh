// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/cu_inc/common.cuh"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
__device__ __inline__ T ComputeGeluGradScalar(T dY, T X, gelu_computation_mode::Default) {
  const T kAlpha = T(M_2_SQRTPI) * T(M_SQRT1_2) * T(0.5);
  return dY * _Fma(X * kAlpha, _Exp(-T(0.5) * X * X), _Normcdf(X));
}

template <typename T>
__device__ __inline__ T ComputeGeluGradScalar(T dY, T X, gelu_computation_mode::Approximation) {
  // copied and adapted from DeepSpeed:
  // https://github.com/microsoft/DeepSpeed/blob/f5025506de37f617a93eabc2aed7cc4f4bfd7d80/csrc/transformer/gelu_kernels.cu#L10

  const float X_float = static_cast<float>(X);

  const float sqrt_param = 0.79788456080286535587989211986876f;
  const float mul_param = 0.044715f;

  constexpr float one = 1.0;
  constexpr float two = 2.0;

  float x2mul = X_float * X_float * mul_param;

  // float tan_h = tanhf(sqrt_param * (X_float + X_float * x2mul));
  float u = two * sqrt_param * _Fma(X_float, x2mul, X_float);
  float emu = __expf(-u);
  float tan_h = two / (one + emu) - one;

  float dg1 = _Fma(0.5f, tan_h, 0.5f);
  float dg2_scaler = X_float * 0.5f * sqrt_param;
  float dg2 = _Fma(dg2_scaler, -tan_h * tan_h, dg2_scaler);
  float dg3 = dg2 * 3 * x2mul;

  return dY * static_cast<T>(dg1 + dg2 + dg3);
}

}  // namespace rocm
}  // namespace onnxruntime
