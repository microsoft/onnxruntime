// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
  return !__hisinf(value) && !__hisnan(value);
}

}  // namespace hip
}  // namespace onnxruntime