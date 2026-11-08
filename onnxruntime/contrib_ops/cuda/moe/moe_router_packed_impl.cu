// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_router_packed_impl.h"

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

template <typename CudaT>
__global__ void SigmoidScoresKernel(
    const CudaT* input, const CudaT* weight, int width, int experts, float* scores) {
  const int row = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int expert = blockIdx.y * 8 + warp;
  if (expert >= experts) return;
  const CudaT* x = input + static_cast<int64_t>(row) * width;
  const CudaT* w = weight + static_cast<int64_t>(expert) * width;
  float sum = 0.0f;
  for (int offset = lane * 8; offset < width; offset += 256) {
    const uint4 xv = LoadPacked8(x + offset);
    const uint4 wv = LoadPacked8(w + offset);
    const CudaT* xp = reinterpret_cast<const CudaT*>(&xv);
    const CudaT* wp = reinterpret_cast<const CudaT*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sum += gguf_cuda::CudaTypeToFloat<CudaT>(xp[j]) * gguf_cuda::CudaTypeToFloat<CudaT>(wp[j]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) scores[static_cast<int64_t>(row) * experts + expert] = 1.0f / (1.0f + expf(-sum));
}

__device__ __forceinline__ uint32_t SigmoidScoreRank(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 1u;
  if (magnitude == 0) return 0x80000000u;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename Rank>
__device__ __forceinline__ void SelectSigmoidRoutes(
    Rank (&choices)[16], const float (&scores)[16], int k, float scale, Rank empty,
    int32_t* indices, float* weights) {
  const int lane = threadIdx.x;
  int selected = -1;
  float selected_score = 0.0f;
  float sum = 0.0f;
  for (int rank = 0; rank < k; ++rank) {
    Rank best = empty;
    float best_score = 0.0f;
    int best_expert = -1;
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
      if (choices[slot] > best) {
        best = choices[slot];
        best_score = scores[slot];
        best_expert = lane + 32 * slot;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const Rank other = __shfl_xor_sync(0xffffffffu, best, offset);
      const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, offset);
      if (other > best || (other == best && other_expert >= 0 &&
                           (best_expert < 0 || other_expert < best_expert))) {
        best = other;
        best_expert = other_expert;
      }
    }
    const int winner_lane = best_expert & 31;
    const int winner_slot = best_expert >> 5;
    if (best_expert >= 0 && winner_lane == lane) {
#pragma unroll
      for (int slot = 0; slot < 16; ++slot) {
        if (slot == winner_slot) choices[slot] = empty;
      }
    }
    const float score = best_expert >= 0 ? __shfl_sync(0xffffffffu, best_score, winner_lane) : 0.0f;
    if (lane == rank) {
      selected = best_expert;
      selected_score = score;
    }
    // Preserve selected-rank FP32 normalization and post-normalization scaling.
    sum += score;
  }
  if (lane < k) {
    indices[lane] = selected;
    weights[lane] = selected_score / (sum + 1e-20f) * scale;
  }
}

__global__ void SigmoidTopKKernel(
    const float* probabilities, const float* bias, int experts, int k, float scale,
    int32_t* indices, float* weights) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  float choices[16];
  float scores[16];
  bool finite = true;
#pragma unroll
  for (int slot = 0; slot < 16; ++slot) {
    const int expert = lane + 32 * slot;
    scores[slot] = expert < experts ? probabilities[static_cast<int64_t>(row) * experts + expert] : 0.0f;
    choices[slot] = expert < experts ? scores[slot] + bias[expert] : -INFINITY;
    finite = finite && (expert >= experts || isfinite(choices[slot]));
  }
  indices += static_cast<int64_t>(row) * k;
  weights += static_cast<int64_t>(row) * k;
  if (__all_sync(0xffffffffu, finite)) {
    SelectSigmoidRoutes(choices, scores, k, scale, -INFINITY, indices, weights);
    return;
  }
  uint32_t ranks[16];
#pragma unroll
  for (int slot = 0; slot < 16; ++slot) {
    ranks[slot] = lane + 32 * slot < experts ? SigmoidScoreRank(choices[slot]) : 0u;
  }
  SelectSigmoidRoutes(ranks, scores, k, scale, 0u, indices, weights);
}

}  // namespace

Status LaunchMoERouterPackedCuda(
    cudaStream_t stream, bool is_bf16, int rows, int width, int experts, int k, float scale,
    const void* input, const void* weight, const float* bias,
    float* scores, int32_t* indices, float* weights) {
  if (rows == 0) return Status::OK();
  const dim3 grid(rows, (experts + 7) / 8);
  if (is_bf16) {
    SigmoidScoresKernel<__nv_bfloat16><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<const __nv_bfloat16*>(weight),
        width, experts, scores);
  } else {
    SigmoidScoresKernel<half><<<grid, 256, 0, stream>>>(
        static_cast<const half*>(input), static_cast<const half*>(weight),
        width, experts, scores);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  SigmoidTopKKernel<<<rows, 32, 0, stream>>>(scores, bias, experts, k, scale, indices, weights);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
