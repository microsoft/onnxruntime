// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_group_router_impl.h"

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
    if constexpr (sizeof(CudaT) == 2) {
      const uint4 xv = LoadPacked8(x + offset);
      const uint4 wv = LoadPacked8(w + offset);
      const CudaT* xp = reinterpret_cast<const CudaT*>(&xv);
      const CudaT* wp = reinterpret_cast<const CudaT*>(&wv);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sum += gguf_cuda::CudaTypeToFloat<CudaT>(xp[j]) * gguf_cuda::CudaTypeToFloat<CudaT>(wp[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sum += x[offset + j] * w[offset + j];
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) scores[static_cast<int64_t>(row) * experts + expert] = 1.0f / (1.0f + expf(-sum));
}

template <typename T>
__global__ void SigmoidLogitsKernel(const T* logits, int64_t count, float* scores) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) scores[i] = 1.0f / (1.0f + expf(-static_cast<float>(logits[i])));
}

__device__ __forceinline__ uint32_t SigmoidScoreRank(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 1u;
  if (magnitude == 0) return 0x80000000u;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename Rank, typename T>
__device__ __forceinline__ void SelectSigmoidRoutes(
    Rank (&choices)[16], const float (&scores)[16], int k, float scale, Rank empty,
    int32_t* indices, float* weights, T* log_scores) {
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
    if (log_scores) log_scores[selected] = static_cast<T>(logf(fmaxf(selected_score, 1e-20f)));
  }
}

template <typename T>
__global__ void SigmoidTopKKernel(
    const float* probabilities, const float* bias, int experts, int k, float scale,
    int32_t* indices, float* weights, T* log_scores, float sentinel) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  float choices[16];
  float scores[16];
  bool finite = true;
  if (log_scores) log_scores += static_cast<int64_t>(row) * experts;
#pragma unroll
  for (int slot = 0; slot < 16; ++slot) {
    const int expert = lane + 32 * slot;
    scores[slot] = expert < experts ? probabilities[static_cast<int64_t>(row) * experts + expert] : 0.0f;
    choices[slot] = expert < experts ? scores[slot] + bias[expert] : -INFINITY;
    finite = finite && (expert >= experts || isfinite(choices[slot]));
    if (log_scores && expert < experts) log_scores[expert] = static_cast<T>(sentinel);
  }
  indices += static_cast<int64_t>(row) * k;
  weights += static_cast<int64_t>(row) * k;
  if (log_scores) __syncwarp();
  if (__all_sync(0xffffffffu, finite)) {
    SelectSigmoidRoutes(choices, scores, k, scale, -INFINITY, indices, weights, log_scores);
    return;
  }
  uint32_t ranks[16];
#pragma unroll
  for (int slot = 0; slot < 16; ++slot) {
    ranks[slot] = lane + 32 * slot < experts ? SigmoidScoreRank(choices[slot]) : 0u;
  }
  SelectSigmoidRoutes(ranks, scores, k, scale, 0u, indices, weights, log_scores);
}

}  // namespace

template <typename T>
Status LaunchCompactMoEGroupRouter(
    cudaStream_t stream, int rows, int width, int experts, int k, float scale,
    const T* input, const T* projection_weight, const float* bias,
    float* scores, int32_t* indices, float* weights, T* log_scores, float sentinel) {
  if (rows == 0) return Status::OK();
  if (projection_weight) {
    const dim3 grid(rows, (experts + 7) / 8);
    SigmoidScoresKernel<T><<<grid, 256, 0, stream>>>(input, projection_weight, width, experts, scores);
  } else {
    const int64_t count = static_cast<int64_t>(rows) * experts;
    SigmoidLogitsKernel<T><<<(count + 255) / 256, 256, 0, stream>>>(input, count, scores);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  SigmoidTopKKernel<T><<<rows, 32, 0, stream>>>(scores, bias, experts, k, scale, indices, weights, log_scores, sentinel);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchCompactMoEGroupRouter<float>(
    cudaStream_t, int, int, int, int, float, const float*, const float*, const float*,
    float*, int32_t*, float*, float*, float);
template Status LaunchCompactMoEGroupRouter<half>(
    cudaStream_t, int, int, int, int, float, const half*, const half*, const float*,
    float*, int32_t*, float*, half*, float);
template Status LaunchCompactMoEGroupRouter<__nv_bfloat16>(
    cudaStream_t, int, int, int, int, float, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    float*, int32_t*, float*, __nv_bfloat16*, float);

Status LaunchProjectedSigmoidRouting(
    cudaStream_t stream, bool is_bf16, int rows, int width, int experts, int k, float scale,
    const void* input, const void* weight, const float* bias,
    float* scores, int32_t* indices, float* weights) {
  if (is_bf16) {
    return LaunchCompactMoEGroupRouter<__nv_bfloat16>(
        stream, rows, width, experts, k, scale, static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight), bias, scores, indices, weights, nullptr, -10000.f);
  }
  return LaunchCompactMoEGroupRouter<half>(
      stream, rows, width, experts, k, scale, static_cast<const half*>(input),
      static_cast<const half*>(weight), bias, scores, indices, weights, nullptr, -10000.f);
}

}  // namespace onnxruntime::contrib::cuda
