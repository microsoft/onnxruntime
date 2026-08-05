// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hash_router.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float ToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half FromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T, typename I>
__global__ void HashRouterKernel(
    T* routing_weights, I* expert_indices, const T* logits, const I* input_ids,
    const I* token_to_expert, int num_experts, int selected_count, int score_function,
    float scaling_factor, float epsilon) {
  if (threadIdx.x != 0) return;
  const int token = blockIdx.x;
  const int64_t token_id = static_cast<int64_t>(input_ids[token]);
  float denominator = 0.0f;
  for (int index = 0; index < selected_count; ++index) {
    const I expert = token_to_expert[token_id * selected_count + index];
    expert_indices[token * selected_count + index] = expert;
    const float logit = ToFloat(logits[token * num_experts + static_cast<int>(expert)]);
    const float score = score_function == 0
                            ? 1.0f / (1.0f + expf(-logit))
                            : sqrtf(fmaxf(0.0f, log1pf(expf(-fabsf(logit))) + fmaxf(logit, 0.0f)));
    routing_weights[token * selected_count + index] = FromFloat<T>(score);
    denominator += score;
  }
  denominator = fmaxf(denominator, epsilon);
  for (int index = 0; index < selected_count; ++index) {
    const int offset = token * selected_count + index;
    routing_weights[offset] = FromFloat<T>(ToFloat(routing_weights[offset]) * scaling_factor / denominator);
  }
}

static int ChooseBlockSize(int head_size, int max_threads) {
  int threads = std::min(head_size, max_threads);
  int power = 1;
  while (power * 2 <= threads) power *= 2;
  return power;
}

template <typename T, typename I>
Status LaunchHashRouterKernel(
    cudaStream_t stream, T* routing_weights, I* expert_indices, const T* logits,
    const I* input_ids, const I* token_to_expert, int token_count, int num_experts,
    int selected_count, int, int score_function, float scaling_factor,
    float epsilon, int) {
  HashRouterKernel<T, I><<<token_count, 1, 0, stream>>>(
      routing_weights, expert_indices, logits, input_ids, token_to_expert, num_experts,
      selected_count, score_function, scaling_factor, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchHashRouterKernel<half, int32_t>(cudaStream_t, half*, int32_t*, const half*, const int32_t*, const int32_t*, int, int, int, int, int, float, float, int); template Status LaunchHashRouterKernel<half, int64_t>(cudaStream_t, half*, int64_t*, const half*, const int64_t*, const int64_t*, int, int, int, int, int, float, float, int); template Status LaunchHashRouterKernel<BFloat16, int32_t>(cudaStream_t, BFloat16*, int32_t*, const BFloat16*, const int32_t*, const int32_t*, int, int, int, int, int, float, float, int); template Status LaunchHashRouterKernel<BFloat16, int64_t>(cudaStream_t, BFloat16*, int64_t*, const BFloat16*, const int64_t*, const int64_t*, int, int, int, int, int, float, float, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
