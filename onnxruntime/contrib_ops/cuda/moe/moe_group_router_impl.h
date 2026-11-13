// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

#ifdef __CUDACC__
template <typename CudaT>
__device__ __forceinline__ uint4 LoadPacked8(const CudaT* input) {
  if ((reinterpret_cast<uintptr_t>(input) & 15u) == 0) {
    return __ldg(reinterpret_cast<const uint4*>(input));
  }
  uint4 result;
  CudaT* values = reinterpret_cast<CudaT*>(&result);
#pragma unroll
  for (int j = 0; j < 8; ++j) values[j] = input[j];
  return result;
}
#endif

template <typename T>
Status LaunchFullMoEGroupRouter(
    cudaStream_t stream, int rows, int width, int experts,
    const T* input, const T* projection_weight, const float* bias,
    float* selection_scores, float* router_weights,
    cublasHandle_t cublas = nullptr);

}  // namespace onnxruntime::contrib::cuda
