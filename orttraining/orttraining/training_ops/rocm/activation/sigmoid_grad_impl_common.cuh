// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/cu_inc/common.cuh"

namespace onnxruntime {
namespace rocm {

template <typename T>
__device__ __inline__ T ComputeSigmoidGradScalar(T dY, T Y) {
  return dY * Y * (1.0 - Y);
}
