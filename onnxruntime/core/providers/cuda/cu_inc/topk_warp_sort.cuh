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

#include "core/providers/cuda/cu_inc/cub.cuh"

namespace onnxruntime {
namespace cuda {
namespace topk_warp {

constexpr int kWarpSize = 32;

// Compile-time threshold guidance based on the onnxruntime-genai offline
// benchmark (NVIDIA H200, CUDA 12.8). Use WarpBitonicSortDescending for sort
// sizes up to kWarpBitonicMaxSize, and the CUB warp merge sort for sizes up to
// kWarpMergeMaxSize. Larger sizes should fall back to a block-wide sort.
constexpr int kWarpBitonicMaxSize = 32;
constexpr int kWarpMergeMaxSize = 64;

/**
 * @brief In-register, warp-wide bitonic sort of kWarpSize (32) (score, index)
 *        pairs, producing a descending order.
 *
 * Each lane in the warp contributes exactly one (score, index) pair. After the
 * call, the warp's pairs are sorted so that lane 0 holds the largest score.
 * Ties on the score are broken in favor of the smaller index. Data is exchanged
 * with __shfl_sync, so no shared memory is required.
 *
 * Lanes that do not hold a valid element should pass score = -FLT_MAX and
 * index = INT_MAX so that they sort to the bottom.
 */
__device__ inline void WarpBitonicSortDescending(float& score, int& index) {
  const int lane_id = threadIdx.x % kWarpSize;

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

// Composite key sorted by the CUB warp merge sort. Sorting the (score, index)
// pair as a single key lets the comparator break ties deterministically without
// relying on the sort being stable (CUB merge sort is not stable).
struct ScoreIndex {
  float score;
  int index;
};

// Descending comparator with smaller-index tie-breaking.
struct ScoreIndexGreater {
  __device__ __forceinline__ bool operator()(const ScoreIndex& a, const ScoreIndex& b) const {
    return a.score > b.score || (a.score == b.score && a.index < b.index);
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
  using SortT = cub::WarpMergeSort<ScoreIndex, kItemsPerThread, kWarpSize>;
  using TempStorage = typename SortT::TempStorage;

  // num_valid_items elements are read from shared memory; the remainder are
  // padded with (-FLT_MAX, INT_MAX) so they sort to the bottom.
  __device__ static void Sort(float* smem_scores, int* smem_indices,
                              TempStorage& temp_storage, int num_valid_items) {
    if (threadIdx.x >= kWarpSize) {
      return;
    }

    ScoreIndex items[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int idx = threadIdx.x + i * kWarpSize;
      if (idx < num_valid_items) {
        items[i].score = smem_scores[idx];
        items[i].index = smem_indices[idx];
      } else {
        items[i].score = -FLT_MAX;
        items[i].index = INT_MAX;
      }
    }

    SortT(temp_storage).Sort(items, ScoreIndexGreater());

    // Blocked write-back: rank r lives at smem[r].
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      const int idx = threadIdx.x * kItemsPerThread + i;
      if (idx < BufferSize) {
        smem_scores[idx] = items[i].score;
        smem_indices[idx] = items[i].index;
      }
    }
  }
};

}  // namespace topk_warp
}  // namespace cuda
}  // namespace onnxruntime
