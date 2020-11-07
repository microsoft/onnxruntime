// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "orttraining/training_ops/rocm/math/isfinite.h"

namespace onnxruntime {
namespace rocm {

template<typename T>
__device__ __forceinline__ bool _IsFiniteScalar(const T value) {
  return isfinite(value);
}

template<>
__device__ __forceinline__ bool _IsFiniteScalar(const half value) {
  return !__hisinf(value) && !__hisnan(value);
}

}  // namespace rocm
}  // namespace onnxruntime