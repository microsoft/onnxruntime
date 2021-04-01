// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/math/isfinite.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value) {
  return isfinite(value);
}

template <typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value, const bool isinf_only, const bool isnan_only) {
  if (isinf_only) {
    return !isinf(value);
  } else if (isnan_only) {
    return !isnan(value);
  } else {
    return isfinite(value);
  }
}

template <>
__device__ __forceinline__ bool _IsFiniteScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return !__hisinf(value) && !__hisnan(value);
#else
  return isfinite(float(value));
#endif
}

template <>
__device__ __forceinline__ bool _IsFiniteScalar(const half value, const bool isinf_only, const bool isnan_only) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  if (isinf_only) {
    return !__hisinf(value);
  } else if (isnan_only) {
    return !__hisnan(value);
  } else {
    return !__hisinf(value) && !__hisnan(value);
  }
#else
  if (isinf_only) {
    return !isinf(float(value));
  } else if (isnan_only) {
    return !isnan(float(value));
  } else {
    return isfinite(float(value));
  }
#endif
}

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
__device__ __forceinline__ bool _IsFiniteScalar(const nv_bfloat16 value) {
  return isfinite(float(value));
}

template <>
__device__ __forceinline__ bool _IsFiniteScalar(const nv_bfloat16 value, const bool isinf_only, const bool isnan_only) {
  if (isinf_only) {
    return !isinf(float(value));
  } else if (isnan_only) {
    return !isnan(float(value));
  } else {
    return isfinite(float(value));
  }
}
#endif

}  // namespace cuda
}  // namespace onnxruntime