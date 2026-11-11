// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace onnxruntime::contrib::cuda {

// Scores determine selection only; unbiased mixing values determine aggregation.
#define DECLARE_FULL_SCORE_ROUTING(T)                                                \
  cudaError_t LaunchFullScoreRouting(const T* scores, const T* mixing, int* indices, \
                                     float* weights, int rows, int experts, int k,   \
                                     bool normalize, cudaStream_t stream);

DECLARE_FULL_SCORE_ROUTING(float)
DECLARE_FULL_SCORE_ROUTING(half)
DECLARE_FULL_SCORE_ROUTING(__nv_bfloat16)

#undef DECLARE_FULL_SCORE_ROUTING

}  // namespace onnxruntime::contrib::cuda
