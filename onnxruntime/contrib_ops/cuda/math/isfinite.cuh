// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/isfinite.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ __forceinline__ bool IsFiniteScalar(const T value) {
  return isfinite(value);
}

template <typename T>
__device__ __forceinline__ bool IsInfScalar(const T value) {
  return isinf(value);
}

template <typename T>
__device__ __forceinline__ bool IsNaNScalar(const T value) {
  return isnan(value);
}

template <>
__device__ __forceinline__ bool IsFiniteScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return !__hisinf(value) && !__hisnan(value);
#else
  return isfinite(float(value));
#endif
}

template <>
__device__ __forceinline__ bool IsInfScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hisinf(value);
#else
  return isinf(float(value));
#endif
}

template <>
__device__ __forceinline__ bool IsNaNScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hisnan(value);
#else
  return isnan(float(value));
#endif
}

template <>
__device__ __forceinline__ bool IsFiniteScalar(const BFloat16 value) {
  return isfinite(static_cast<float>(value));
}

template <>
__device__ __forceinline__ bool IsInfScalar(const BFloat16 value) {
  return isinf(static_cast<float>(value));
}

template <>
__device__ __forceinline__ bool IsNaNScalar(const BFloat16 value) {
  return isnan(static_cast<float>(value));
}

}  // namespace cuda
}  // namespace onnxruntime