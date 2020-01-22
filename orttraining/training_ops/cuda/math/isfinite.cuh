// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "isfinite.h"

namespace onnxruntime {
namespace cuda {

template<typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value) {
  return isfinite(value);
}

template<>
__device__ __forceinline__ bool _IsFiniteScalar(const half value) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return !__hisinf(value) && !__hisnan(value);
#else
  return isfinite(float(value));
#endif
}

}  // namespace cuda
}  // namespace onnxruntime