// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_full_score_routing.h"

#include <cstdint>

namespace onnxruntime::contrib::cuda {
namespace {

__device__ __forceinline__ uint32_t FullScoreRank(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 1u;
  if (magnitude == 0) return 0x80000000u;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename T>
__global__ __launch_bounds__(256) void FullScoreRoutingKernel(
    const T* selection_scores, const T* mixing, int* output_indices, float* output_weights,
    int rows, int experts, int k, bool normalize) {
  const int row = static_cast<int>(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  uint32_t choices[16];
  for (int slot = 0; slot < 16; ++slot) {
    const int expert = lane + 32 * slot;
    choices[slot] = expert < experts
                        ? FullScoreRank(static_cast<float>(selection_scores[static_cast<int64_t>(row) * experts + expert]))
                        : 0u;
  }
  int selected = 0;
  float selected_weight = 0.0f;
  float sum = 0.0f;
  bool finite = true;
  for (int rank = 0; rank < k; ++rank) {
    uint32_t best = 0u;
    int expert = 512;
    for (int slot = 0; slot < 16; ++slot) {
      if (choices[slot] > best) {
        best = choices[slot];
        expert = lane + 32 * slot;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const uint32_t other = __shfl_xor_sync(0xffffffffu, best, offset);
      const int other_expert = __shfl_xor_sync(0xffffffffu, expert, offset);
      if (other > best || (other == best && other_expert < expert)) {
        best = other;
        expert = other_expert;
      }
    }
    if ((expert & 31) == lane) choices[expert >> 5] = 0u;
    const float weight = static_cast<float>(mixing[static_cast<int64_t>(row) * experts + expert]);
    finite = finite && isfinite(weight);
    sum += weight;
    if (lane == rank) {
      selected = expert;
      selected_weight = weight;
    }
  }
  if (lane < k) {
    const float inverse_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
    const float weight = normalize ? selected_weight * inverse_sum : selected_weight;
    output_indices[static_cast<int64_t>(row) * k + lane] = selected;
    output_weights[static_cast<int64_t>(row) * k + lane] =
        finite && isfinite(weight) ? weight : __int_as_float(0x7fffffff);
  }
}

// QMoE also permits expert counts and k outside the warp-cached path's limits.
template <typename T>
__global__ void FullScoreRoutingGenericKernel(
    const T* scores, const T* mixing, int* indices, float* weights,
    int rows, int experts, int k, bool normalize) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  uint32_t previous = 0xffffffffu;
  int previous_expert = -1;
  float sum = 0.0f;
  bool finite = true;
  for (int rank = 0; rank < k; ++rank) {
    uint32_t best = 0u;
    int selected = experts;
    for (int expert = 0; expert < experts; ++expert) {
      const uint32_t value = FullScoreRank(static_cast<float>(scores[row * experts + expert]));
      if ((value < previous || (value == previous && expert > previous_expert)) &&
          (value > best || (value == best && expert < selected))) {
        best = value;
        selected = expert;
      }
    }
    previous = best;
    previous_expert = selected;
    const float weight = static_cast<float>(mixing[row * experts + selected]);
    indices[row * k + rank] = selected;
    weights[row * k + rank] = weight;
    finite = finite && isfinite(weight);
    sum += weight;
  }
  const float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
  for (int rank = 0; rank < k; ++rank) {
    const float weight = normalize ? weights[row * k + rank] * inverse : weights[row * k + rank];
    weights[row * k + rank] = finite && isfinite(weight) ? weight : __int_as_float(0x7fffffff);
  }
}

template <typename T>
cudaError_t DispatchFullScoreRouting(const T* scores, const T* mixing, int* indices, float* weights,
                                     int rows, int experts, int k, bool normalize, cudaStream_t stream) {
  if (rows < 0 || k <= 0 || k > experts) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (experts <= 512 && k <= 32) {
    FullScoreRoutingKernel<<<(rows - 1) / 8 + 1, 256, 0, stream>>>(
        scores, mixing, indices, weights, rows, experts, k, normalize);
  } else {
    FullScoreRoutingGenericKernel<<<(rows - 1) / 128 + 1, 128, 0, stream>>>(
        scores, mixing, indices, weights, rows, experts, k, normalize);
  }
  return cudaGetLastError();
}

}  // namespace

#define DEFINE_FULL_SCORE_ROUTING(T)                                                 \
  cudaError_t LaunchFullScoreRouting(const T* scores, const T* mixing, int* indices, \
                                     float* weights, int rows, int experts, int k,   \
                                     bool normalize, cudaStream_t stream) {          \
    return DispatchFullScoreRouting(scores, mixing, indices, weights,                \
                                    rows, experts, k, normalize, stream);            \
  }

DEFINE_FULL_SCORE_ROUTING(float)
DEFINE_FULL_SCORE_ROUTING(half)
DEFINE_FULL_SCORE_ROUTING(__nv_bfloat16)

#undef DEFINE_FULL_SCORE_ROUTING

}  // namespace onnxruntime::contrib::cuda
