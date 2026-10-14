// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Adapted from TLWU 3e09dced1b with 1f9387d872 guards; no dependent launches.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <type_traits>
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime::contrib::cuda {

__device__ __forceinline__ uint32_t FusedRouteKey(float logit, int index) {
  const uint32_t bits = logit == 0.0f ? 0u : __float_as_uint(logit);
  const uint32_t sortable = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return (sortable & 0xFFFFE000u) | (0x1FFFu - static_cast<uint32_t>(index));
}

__device__ __forceinline__ float FusedRouteScore(uint32_t key) {
  const uint32_t sortable = key & 0xFFFFE000u;
  return __uint_as_float((sortable & 0x80000000u) ? (sortable & 0x7FFFFFFFu) : (~sortable & 0xFFFFE000u));
}

template <typename T>
__device__ __forceinline__ void FusedRouteLoad4(const T* p, float (&v)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  if constexpr (std::is_same_v<T, half>) {
    const half2 a = *reinterpret_cast<const half2*>(&raw.x);
    const half2 b = *reinterpret_cast<const half2*>(&raw.y);
    v[0] = __low2float(a); v[1] = __high2float(a);
    v[2] = __low2float(b); v[3] = __high2float(b);
  } else {
    v[0] = __uint_as_float(raw.x << 16); v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
    v[2] = __uint_as_float(raw.y << 16); v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
  }
}

// One warp emulates the four warps of SoftmaxTopKMergeKernel<T,128,4>,
// retaining its exp-sum tree and sequential top-k normalization order.
template <typename T, int K>
__global__ __launch_bounds__(128) void FusedSoftmaxTopKMapsKernel(
    const T* logits, float* scales, int* indices, int* r2u, int* u2r,
    int* permuted_expert, int64_t* offsets, int rows, bool normalize) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ int keys_selected[4 * 16];
  __shared__ int sorted[128];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  sorted[threadIdx.x] = INT_MAX;
  if (warp < rows) {
    float scores[16], maximum = onnxruntime::cuda::topk::kNegativeInfinity;
    uint32_t keys[16];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int base = (lane + 32 * w) * 4;
      float v[4]; FusedRouteLoad4(logits + warp * 512 + base, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        scores[w * 4 + j] = v[j] == 0.0f ? 0.0f : v[j];
        keys[w * 4 + j] = FusedRouteKey(v[j], base + j);
        maximum = fmaxf(maximum, v[j]);
      }
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, maximum, offset);
      maximum = maximum < other ? other : maximum;
    }
    uint32_t winners[K];
#pragma unroll
    for (int r = 0; r < K; ++r) {
      uint32_t best = keys[0];
#pragma unroll
      for (int j = 1; j < 16; ++j) best = max(best, keys[j]);
      winners[r] = __reduce_max_sync(0xFFFFFFFFu, best);
#pragma unroll
      for (int j = 0; j < 16; ++j) keys[j] = keys[j] == winners[r] ? 0u : keys[j];
    }
    float part[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) sum += expf(scores[w * 4 + j] - maximum);
      part[w] = sum;
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
      for (int w = 0; w < 4; ++w) part[w] += __shfl_xor_sync(0xFFFFFFFFu, part[w], offset);
    }
    const float sum = ((__shfl_sync(0xFFFFFFFFu, part[0], 0) + __shfl_sync(0xFFFFFFFFu, part[1], 0)) +
                       __shfl_sync(0xFFFFFFFFu, part[2], 0)) + __shfl_sync(0xFFFFFFFFu, part[3], 0);
    const float inv_sum = onnxruntime::cuda::topk::SafeInvSum(sum);
    uint32_t selected = 0; float probability = 0, scale_sum = 0;
#pragma unroll
    for (int r = 0; r < K; ++r) {
      const float p = onnxruntime::cuda::topk::SoftmaxScale(FusedRouteScore(winners[r]), maximum, inv_sum);
      scale_sum += p;
      if (lane == r) { selected = winners[r]; probability = p; }
    }
    const float denom = onnxruntime::cuda::topk::TopKNormalizeDenom(normalize, scale_sum);
    if (lane < K) {
      const int expert = 0x1FFF - static_cast<int>(selected & 0x1FFFu);
      scales[warp * K + lane] = normalize ? probability / denom : probability;
      indices[warp * K + lane] = expert;
      keys_selected[warp * K + lane] = (expert << 2) | warp;
    }
  }
  __syncthreads();
  const int pairs = rows * K;
  if (threadIdx.x < pairs) {
    const int p = threadIdx.x, key = keys_selected[p];
    int rank = 0;
    for (int q = 0; q < pairs; ++q) rank += keys_selected[q] < key;
    const int token = key & 3, expert = key >> 2;
    const int source = (p - token * K) * rows + token;
    r2u[rank] = source; u2r[source] = rank; permuted_expert[rank] = expert;
    sorted[rank] = expert;
  }
  __syncthreads();
  for (int expert = threadIdx.x; expert <= 512; expert += blockDim.x) {
    int lo = 0, hi = pairs;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (sorted[mid] < expert) lo = mid + 1; else hi = mid;
    }
    offsets[expert] = lo;
  }
#endif
}

template <typename T>
bool LaunchFusedRouteMapsImpl(const T* logits, float* scales, int* indices,
                          int* r2u, int* u2r, int* permuted_expert, int64_t* offsets,
                          int rows, int experts, int k, bool normalize, cudaStream_t stream) {
  if (experts != 512 || rows < 1 || rows > 4 ||
      (reinterpret_cast<uintptr_t>(logits) % 8) != 0) return false;
#define LAUNCH_ROUTE(K) \
  FusedSoftmaxTopKMapsKernel<T, K><<<1, 128, 0, stream>>>( \
      logits, scales, indices, r2u, u2r, permuted_expert, offsets, rows, normalize)
  switch (k) {
    case 1: LAUNCH_ROUTE(1); break;
    case 2: LAUNCH_ROUTE(2); break;
    case 4: LAUNCH_ROUTE(4); break;
    case 6: LAUNCH_ROUTE(6); break;
    case 8: LAUNCH_ROUTE(8); break;
    case 10: LAUNCH_ROUTE(10); break;
    default: return false;
  }
#undef LAUNCH_ROUTE
  return true;
}
}  // namespace onnxruntime::contrib::cuda
