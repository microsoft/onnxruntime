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

__device__ __forceinline__ int LaneId() {
  int lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ int LinearThreadIdInBlock() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
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

// Convert a (score, index) pair into a single unsigned integer key. Descending
// integer order then gives descending float score order, with equal scores
// preferring the smaller original index. This matches the stable Top-K packing
// used by onnxruntime-genai while avoiding a compound comparator in CUB.
__device__ __forceinline__ uint64_t PackStableSortKey(float score, int index) {
  const uint32_t score_bits = __float_as_uint(score);
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

  // num_valid_items elements are read from shared memory; the remainder are
  // padded with (kNegativeInfinity, INT_MAX) so valid -inf scores sort ahead of padding.
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
        items[i] = PackStableSortKey(kNegativeInfinity, INT_MAX);
      }
    }

    SortT(temp_storage).Sort(items, Greater<uint64_t>());

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
