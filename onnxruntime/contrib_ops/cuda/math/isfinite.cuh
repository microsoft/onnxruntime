// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/isfinite.h"

#if CUDA_VERSION >= 11000
#include "cuda_bf16.h"
#endif

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

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__device__ __forceinline__ bool IsFiniteScalar(const nv_bfloat16 value) {
  return !__hisinf(value) && !__hisnan(value);
}

template <>
__device__ __forceinline__ bool IsInfScalar(const nv_bfloat16 value) {
  return __hisinf(value);
}

template <>
__device__ __forceinline__ bool IsNaNScalar(const nv_bfloat16 value) {
  return __hisnan(value);
}
#endif

}  // namespace cuda
}  // namespace onnxruntime