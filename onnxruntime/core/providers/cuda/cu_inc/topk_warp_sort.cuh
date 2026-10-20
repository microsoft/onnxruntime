// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Reusable warp-level Top-K sorting primitives for CUDA.
//
// These helpers sort (score, index) pairs in descending order. Ties on the
// score are broken deterministically by preferring the smaller index, matching
// the tie-breaking used by the onnxruntime-genai Top-K kernels (the
// `STABLE_TOPK` path in cuda_topk_warp_sort_helper.cuh).
//
// Two primitives are provided, mirroring the algorithms that the genai offline
// benchmark found fastest for small sort sizes:
//   * WarpBitonicSortDescending : best for sort sizes up to 32. Each lane holds
//     a single (score, index) pair entirely in registers and exchanges data
//     via warp shuffles, avoiding shared memory.
//   * WarpMergeSorter            : best for sort sizes up to 64 (CUB warp merge
//     sort). A single warp sorts up to `BufferSize` pairs held in shared memory.
//
// They are intentionally operator-agnostic so they can be reused outside the
// MoE Top-K path.

#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>
#include <limits>

#include "core/providers/cuda/cu_inc/cub.cuh"

namespace onnxruntime {
namespace cuda {
namespace topk {

constexpr int kWarpSize = 32;

// Compile-time threshold guidance based on the onnxruntime-genai offline
// benchmark (NVIDIA H200, CUDA 12.8). Use WarpBitonicSortDescending for sort
// sizes up to kWarpBitonicMaxSize, and the CUB warp merge sort for sizes up to
// kWarpMergeMaxSize. Larger sizes should fall back to a block-wide sort.
constexpr int kWarpBitonicMaxSize = 32;
constexpr int kWarpMergeMaxSize = 64;
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
constexpr uint64_t kPaddingSortKey = 0;

__device__ __forceinline__ int LaneId() {
  int lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ int LinearThreadIdInBlock() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

// Softmax / top-k score helpers shared by the MoE routing kernels. They live here (rather than in
// one of the MoE translation units) because two libraries need bit-identical results: the standalone
// SoftmaxTopK kernels in contrib_ops/cuda/moe/qmoe_kernels.cu, and the fused routing prologue in
// contrib_ops/cuda/llm/moe_gemm/moe_kernels.cu. Any divergence would silently change expert scales.
constexpr float kTopKNormalizeEpsilon = 1e-6f;

__device__ __forceinline__ float SoftmaxScale(float logit, float max_val, float inv_sum) {
  return (inv_sum > 0.0f) ? expf(logit - max_val) * inv_sum : 0.0f;
}

__device__ __forceinline__ float SafeInvSum(float sum) {
  return (sum > 0.0f) ? (1.0f / sum) : 0.0f;
}

__device__ __forceinline__ float TopKNormalizeDenom(bool normalize_scales, float scale_sum) {
  return (normalize_scales && scale_sum > kTopKNormalizeEpsilon) ? scale_sum : 1.0f;
}

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

/**
 * @brief In-register, warp-wide bitonic sort of kWarpSize (32) (score, index)
 *        pairs, producing a descending order.
 *
 * Each lane in the warp contributes exactly one (score, index) pair. After the
 * call, the warp's pairs are sorted so that lane 0 holds the largest score.
 * Ties on the score are broken in favor of the smaller index. Data is exchanged
 * with __shfl_sync, so no shared memory is required.
 *
 * Lanes that do not hold a valid element should pass score = kNegativeInfinity
 * and index = INT_MAX so that valid -inf scores sort ahead of padding.
 */
__device__ inline void WarpBitonicSortDescending(float& score, int& index) {
  const int lane_id = LaneId();

  // Build the bitonic sorting network in stages.
  for (int k = 2; k <= kWarpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int paired_lane = lane_id ^ j;
      const float paired_score = __shfl_sync(0xFFFFFFFF, score, paired_lane);
      const int paired_index = __shfl_sync(0xFFFFFFFF, index, paired_lane);

      // A standard bitonic network sorts ascending when (lane_id & k) == 0; we
      // invert the swap condition to produce an overall descending sort.
      const bool direction = ((lane_id & k) == 0);

      // Tie-break: equal scores prefer the smaller index.
      const bool is_mine_greater =
          (score > paired_score) || (score == paired_score && index < paired_index);

      const float s_max = is_mine_greater ? score : paired_score;
      const int i_max = is_mine_greater ? index : paired_index;
      const float s_min = is_mine_greater ? paired_score : score;
      const int i_min = is_mine_greater ? paired_index : index;

      if (direction) {
        score = (lane_id < paired_lane) ? s_max : s_min;
        index = (lane_id < paired_lane) ? i_max : i_min;
      } else {
        score = (lane_id < paired_lane) ? s_min : s_max;
        index = (lane_id < paired_lane) ? i_min : i_max;
      }
    }
  }
}

__device__ inline void WarpBitonicSortDescending(uint64_t& key) {
  const int lane_id = LaneId();

  for (int k = 2; k <= kWarpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int paired_lane = lane_id ^ j;
      const uint64_t paired_key = static_cast<uint64_t>(
          __shfl_sync(0xFFFFFFFFu, static_cast<unsigned long long>(key), paired_lane));
      const bool direction = ((lane_id & k) == 0);
      const uint64_t key_max = key > paired_key ? key : paired_key;
      const uint64_t key_min = key > paired_key ? paired_key : key;

      if (direction) {
        key = (lane_id < paired_lane) ? key_max : key_min;
      } else {
        key = (lane_id < paired_lane) ? key_min : key_max;
      }
    }
  }
}

// Convert a (score, index) pair into a single unsigned integer key. Descending
// integer order then gives descending float score order, with equal scores
// preferring the smaller original index. This matches the stable Top-K packing
// used by onnxruntime-genai while avoiding a compound comparator in CUB.
__device__ __forceinline__ uint64_t PackStableSortKey(float score, int index) {
  const uint32_t score_bits = score == 0.0f ? 0u : __float_as_uint(score);
  const uint32_t sortable_score =
      (score_bits & 0x80000000u) ? (~score_bits) : (score_bits | 0x80000000u);
  const uint32_t inverted_index = UINT_MAX - static_cast<uint32_t>(index);
  return (static_cast<uint64_t>(sortable_score) << 32) | inverted_index;
}

__device__ __forceinline__ float UnpackStableSortScore(uint64_t key) {
  const uint32_t sortable_score = static_cast<uint32_t>(key >> 32);
  const uint32_t score_bits =
      (sortable_score & 0x80000000u) ? (sortable_score & 0x7fffffffu) : ~sortable_score;
  return __uint_as_float(score_bits);
}

__device__ __forceinline__ int UnpackStableSortIndex(uint64_t key) {
  const uint32_t inverted_index = static_cast<uint32_t>(key & 0xffffffffu);
  return static_cast<int>(UINT_MAX - inverted_index);
}

template <typename T>
struct Greater {
  __device__ __host__ __forceinline__ bool operator()(const T& a, const T& b) const {
    return a > b;
  }
};

/**
 * @brief Sort `keys[0..N)` descending with an in-register bitonic network.
 *
 * N must be a power of two. The network is fully unrolled and touches no memory,
 * so the whole sort stays in registers.
 */
template <int N>
__device__ __forceinline__ void BitonicSortRegistersDescending(uint64_t (&keys)[N]) {
  static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two.");
#pragma unroll
  for (int k = 2; k <= N; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        const int partner = i ^ j;
        if (partner > i) {
          const uint64_t a = keys[i];
          const uint64_t b = keys[partner];
          const bool a_first = ((i & k) == 0) == (a > b);
          keys[i] = a_first ? a : b;
          keys[partner] = a_first ? b : a;
        }
      }
    }
  }
}

/**
 * @brief Sort a bitonic sequence `keys[0..N)` descending (half-cleaner cascade).
 *
 * N must be a power of two and the input must already be bitonic.
 */
template <int N>
__device__ __forceinline__ void BitonicCleanDescending(uint64_t (&keys)[N]) {
  static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two.");
#pragma unroll
  for (int j = N >> 1; j > 0; j >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const int partner = i ^ j;
      if (partner > i) {
        const uint64_t a = keys[i];
        const uint64_t b = keys[partner];
        keys[i] = a > b ? a : b;
        keys[partner] = a > b ? b : a;
      }
    }
  }
}

/**
 * @brief Single-warp Top-N over the warp's 32 * N packed keys, entirely in registers.
 *
 * Every lane contributes N packed keys (see PackStableSortKey; the packing makes a
 * plain unsigned comparison equivalent to "larger score first, smaller index on a
 * tie"). On return every lane holds the N largest keys of the whole warp in
 * `keys[0..N)`, sorted descending; a caller that wants the top k <= N simply reads
 * the first k entries. Which N of the 32 * N inputs a lane starts with is
 * irrelevant, so callers should pick the layout that loads best (e.g. strided by
 * warp size for coalescing).
 *
 * The algorithm is a bitonic sort of each lane's own keys followed by a butterfly
 * of bitonic Top-N merges over the warp. Merging two descending sequences A and B
 * as C[i] = max(A[i], B[N-1-i]) yields a bitonic sequence holding exactly the top N
 * of A union B, which the half-cleaner cascade then sorts. Because that construction
 * is symmetric in A and B, a single __shfl_xor_sync per key leaves both partners
 * with the same result, so no shared memory and no barriers are needed.
 */
template <int N>
__device__ __forceinline__ void WarpBitonicTopN(uint64_t (&keys)[N]) {
  BitonicSortRegistersDescending<N>(keys);
#pragma unroll
  for (int step = 1; step < kWarpSize; step <<= 1) {
    uint64_t partner[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      partner[i] = static_cast<uint64_t>(
          __shfl_xor_sync(0xFFFFFFFFu, static_cast<unsigned long long>(keys[i]), step));
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint64_t other = partner[N - 1 - i];
      keys[i] = keys[i] > other ? keys[i] : other;
    }
    BitonicCleanDescending<N>(keys);
  }
}

/**
 * @brief Single-warp CUB merge sort of up to `BufferSize` (score, index) pairs
 *        held in shared memory, producing a descending order.
 *
 * Only the first warp of the calling block performs work; the caller is
 * responsible for any __syncthreads() needed before (to publish the shared
 * memory inputs) and after (to consume the sorted outputs). On return,
 * smem_scores[r]/smem_indices[r] hold the element of rank r (rank 0 == largest).
 *
 * @tparam BufferSize Maximum number of pairs to sort. Must be <= 256.
 */
template <int BufferSize>
struct WarpMergeSorter {
  static_assert(BufferSize > 0 && BufferSize <= 256, "BufferSize must be in (0, 256].");

  static constexpr int kItemsPerThread = (BufferSize + kWarpSize - 1) / kWarpSize;
  using SortT = cub::WarpMergeSort<uint64_t, kItemsPerThread, kWarpSize, cub::NullType>;
  using TempStorage = typename SortT::TempStorage;

  // num_valid_items elements are read from shared memory; the remainder use the minimum
  // packed key so every valid score, including a negative NaN, sorts ahead of padding.
  // `temp_storage` may alias `smem_scores`/`smem_indices` (callers often union them to save
  // shared memory), so the write-back is fenced against CUB's final reads of that storage.
  __device__ static void Sort(float* smem_scores, int* smem_indices,
                              TempStorage& temp_storage, int num_valid_items) {
    const int thread_id = LinearThreadIdInBlock();
    if (thread_id >= kWarpSize) {
      return;
    }

    const int lane_id = thread_id;

    uint64_t items[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int idx = lane_id + i * kWarpSize;
      if (idx < num_valid_items) {
        items[i] = PackStableSortKey(smem_scores[idx], smem_indices[idx]);
      } else {
        items[i] = kPaddingSortKey;
      }
    }

    // The loads above need no barrier: CUB syncs before its first temp_storage write.
    SortT(temp_storage).Sort(items, Greater<uint64_t>());
    // CUB's merge loop ends on a read of temp_storage with no trailing sync.
    __syncwarp();

    // Blocked write-back: rank r lives at smem[r].
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int idx = lane_id * kItemsPerThread + i;
      if (idx < BufferSize) {
        smem_scores[idx] = UnpackStableSortScore(items[i]);
        smem_indices[idx] = UnpackStableSortIndex(items[i]);
      }
    }
  }
};

}  // namespace topk
}  // namespace cuda
}  // namespace onnxruntime
