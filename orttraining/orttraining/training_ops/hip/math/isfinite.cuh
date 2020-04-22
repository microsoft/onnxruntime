// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include "core/providers/hip/cu_inc/common.cuh"
#include "isfinite.h"

namespace onnxruntime {
namespace hip {

template<typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value) {
  return isfinite(value);
}

template<>
__device__ __forceinline__ bool _IsFiniteScalar(const half value) {
#if __HIP_ARCH__ >= 530 || !defined(__HIP_ARCH__)
  return !__hisinf(value) && !__hisnan(value);
#else
  return isfinite(float(value));
#endif
}

}  // namespace hip
}  // namespace onnxruntime