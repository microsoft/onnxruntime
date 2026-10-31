// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Streaming small-K top-K for a long contiguous last axis.
//
// RadixTopK gives one CUDA block to each row and makes ~7 passes over it, so an LLM-sized
// selection (K = 16 over a 248320-wide vocabulary, 8 rows) runs 8 blocks on a 132-SM device and
// measures 1.36 ms on H200. This path instead makes ONE pass with a grid-wide split.
//
// Each warp keeps a 32-entry sorted-descending list, one entry per lane, of 64-bit composite
// keys `(order_key << 32) | ~index`. The composite makes the comparison total: equal values are
// broken towards the lower index, which is the tie order RadixTopK's BIGGER/SMALLER produce. An
// element can only matter if it beats the current K-th entry, so the warp tests that with a
// single ballot and skips the bitonic merge entirely for all but the first few thousand
// elements. Warps merge through shared memory, blocks through a small candidate buffer that a
// second single-warp kernel reduces.
//
// HybridTopK is preferred when supported, but this path also covers its gaps: smallest-element
// selection (largest == 0), dimensions beyond HybridTopK's 256-partition limit (> 851968), and
// grids that HybridTopK rejects because its cooperative reduction cannot satisfy residency.
//
// Restricted to float/half/bfloat16 on the last axis with K <= 32; everything else falls back.

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "topk_impl.h"

namespace onnxruntime {
namespace cuda {
namespace smallk_topk {

constexpr int64_t kMaxK = 32;
constexpr int64_t kMinDimension = 4096;
constexpr int64_t kElementsPerBlock = 16384;
constexpr int kThreads = 256;
constexpr int kMaxBlocksPerRow = 64;

template <typename T>
struct Supported : std::false_type {};
template <>
struct Supported<float> : std::true_type {};
template <>
struct Supported<__half> : std::true_type {};
template <>
struct Supported<BFloat16> : std::true_type {};
template <>
struct Supported<__nv_bfloat16> : std::true_type {};

// Monotone float -> uint32 map: ordering the bit patterns as unsigned integers reproduces the
// float ordering (negatives are inverted, positives get their sign bit set).
__device__ __forceinline__ uint32_t OrderKey(float v) {
  const uint32_t u = v == 0.0f ? 0u : __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(BFloat16 v) { return static_cast<float>(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ uint64_t Composite(T value, int64_t index, int64_t largest) {
  const uint32_t key = OrderKey(ToFloat(value));
  const uint32_t ordered = (1 == largest) ? key : ~key;
  return (static_cast<uint64_t>(ordered) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(index));
}

// Sorts one value per lane into descending order across the warp.
__device__ __forceinline__ uint64_t WarpSortDescending(uint64_t v) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const uint64_t p = __shfl_xor_sync(0xffffffffu, v, j);
      const bool desc_half = (lane & k) == 0;
      const bool keep_larger = ((lane & j) == 0) == desc_half;
      v = keep_larger ? (v > p ? v : p) : (v < p ? v : p);
    }
  }
  return v;
}

// Keeps the 32 largest of two descending warp-sorted sequences (bitonic merge).
__device__ __forceinline__ uint64_t WarpMergeDescending(uint64_t a, uint64_t b) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const uint64_t br = __shfl_xor_sync(0xffffffffu, b, 31);
  uint64_t m = a > br ? a : br;
#pragma unroll
  for (int j = 16; j > 0; j >>= 1) {
    const uint64_t p = __shfl_xor_sync(0xffffffffu, m, j);
    m = ((lane & j) == 0) ? (m > p ? m : p) : (m < p ? m : p);
  }
  return m;
}

template <typename T>
__global__ void PartialTopK(const T* __restrict__ X, uint64_t* __restrict__ candidates,
                            int64_t dimension, int64_t K, int64_t largest, int blocks_per_row) {
  __shared__ uint64_t warp_best[kThreads];

  const int row = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps = kThreads >> 5;
  const T* row_x = X + static_cast<size_t>(row) * dimension;

  const int64_t stride = static_cast<int64_t>(blocks_per_row) * kThreads;
  uint64_t best = 0;
  uint64_t threshold = 0;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
       base - lane < dimension; base += stride) {
    const uint64_t v = base < dimension ? Composite<T>(row_x[base], base, largest) : 0;
    if (__any_sync(0xffffffffu, v > threshold)) {
      best = WarpMergeDescending(best, WarpSortDescending(v));
      threshold = __shfl_sync(0xffffffffu, best, static_cast<int>(K) - 1);
    }
  }

  warp_best[warp * 32 + lane] = best;
  __syncthreads();
  if (warp != 0) {
    return;
  }
  uint64_t m = warp_best[lane];
  for (int w = 1; w < warps; ++w) {
    m = WarpMergeDescending(m, warp_best[w * 32 + lane]);
  }
  if (lane < K) {
    candidates[(static_cast<size_t>(row) * blocks_per_row + blockIdx.x) * K + lane] = m;
  }
}

template <typename T>
__global__ void FinalTopK(const T* __restrict__ X, T* __restrict__ V, int64_t* __restrict__ I,
                          const uint64_t* __restrict__ candidates, int64_t dimension, int64_t K,
                          int total_candidates) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const uint64_t* row_cand = candidates + static_cast<size_t>(row) * total_candidates;

  uint64_t m = 0;
  for (int base = 0; base < total_candidates; base += 32) {
    const uint64_t v = (base + lane) < total_candidates ? row_cand[base + lane] : 0;
    m = WarpMergeDescending(m, WarpSortDescending(v));
  }
  if (lane < K) {
    const int64_t index = static_cast<int64_t>(~static_cast<uint32_t>(m & 0xffffffffu));
    V[static_cast<size_t>(row) * K + lane] = X[static_cast<size_t>(row) * dimension + index];
    I[static_cast<size_t>(row) * K + lane] = index;
  }
}

inline bool IsSupported(const CudaKernel* kernel, int32_t axis, size_t size,
                        int64_t rows, int64_t dimension, int64_t k) {
  // The partial pass maps rows onto grid.y, which is far more constrained than grid.x.
  return axis == static_cast<int32_t>(size) - 1 &&
         k >= 1 && k <= kMaxK &&
         dimension > kMinDimension && dimension <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) &&
         rows >= 1 && rows <= static_cast<int64_t>(kernel->GetDeviceProp().maxGridSize[1]);
}

template <typename T>
Status Run(const CudaKernel* kernel,
           cudaStream_t stream,
           void* alloc_stream,
           const T* input,
           T* output_v,
           int64_t* output_i,
           int64_t rows,
           int64_t dimension,
           int64_t k,
           int64_t largest) {
  int blocks_per_row = static_cast<int>((dimension + kElementsPerBlock - 1) / kElementsPerBlock);
  blocks_per_row = blocks_per_row < 1 ? 1 : blocks_per_row;
  blocks_per_row = blocks_per_row > kMaxBlocksPerRow ? kMaxBlocksPerRow : blocks_per_row;
  const int total_candidates = blocks_per_row * static_cast<int>(k);
  auto candidates = kernel->GetScratchBuffer<uint64_t>(
      SafeInt<size_t>(rows) * total_candidates, alloc_stream);

  const dim3 partial_grid{static_cast<unsigned>(blocks_per_row), static_cast<unsigned>(rows)};
  PartialTopK<T><<<partial_grid, kThreads, 0, stream>>>(
      input, candidates.get(), dimension, k, largest, blocks_per_row);
  FinalTopK<T><<<static_cast<unsigned>(rows), 32, 0, stream>>>(
      input, output_v, output_i, candidates.get(), dimension, k, total_candidates);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace smallk_topk
}  // namespace cuda
}  // namespace onnxruntime
