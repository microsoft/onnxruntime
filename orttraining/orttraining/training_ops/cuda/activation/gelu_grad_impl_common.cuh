// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cpu/activation/gelu_computation_mode.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ __inline__ T ComputeGeluGradScalar(T dY, T X, gelu_computation_mode::Default) {
  const T kAlpha = T(M_2_SQRTPI) * T(M_SQRT1_2) * T(0.5);
  return dY * (_Normcdf(X) + X * kAlpha * _Exp(-T(0.5) * X * X));
}

template <>
__device__ __inline__ half ComputeGeluGradScalar(half dY, half X, gelu_computation_mode::Default) {
  const half kHalf = half(0.5);
  const half kOne = half(1.0);
  const half kAlpha = half(M_SQRT1_2);
  const half kBeta = half(M_2_SQRTPI) * kAlpha * kHalf;
  return dY * (kHalf * (kOne + _Erf(kAlpha * X)) + X * kBeta * _Exp(-kHalf * X * X));
}

template <typename T>
__device__ __inline__ T ComputeGeluGradScalar(T dY, T X, gelu_computation_mode::Approximation) {
  // copied and adapted from DeepSpeed:
  // https://github.com/microsoft/DeepSpeed/blob/f5025506de37f617a93eabc2aed7cc4f4bfd7d80/csrc/transformer/gelu_kernels.cu#L10

  const float X_float = static_cast<float>(X);

  const float sqrt_param = 0.79788456080286535587989211986876f;
  const float mul_param = 0.044715f;

  float x2mul = X_float * X_float * mul_param;
  float tan_h = tanhf(sqrt_param * (X_float + X_float * x2mul));
  float dg1 = 0.5f * (1.0f + tan_h);
  float dg2 = X_float * 0.5f * sqrt_param * (1 - tan_h * tan_h);
  float dg3 = dg2 * 3 * x2mul;

  return dY * static_cast<T>(dg1 + dg2 + dg3);
}

}  // namespace cuda
}  // namespace onnxruntime
