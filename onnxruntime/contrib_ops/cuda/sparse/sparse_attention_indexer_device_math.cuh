// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Device-side math shared by com.microsoft.SparseAttentionIndexer (dense, growing state) and
// com.microsoft.PackedSparseAttentionIndexer (packed, fixed-capacity state). Everything here is a
// small, self-contained helper with no dependency on either operator's tensor layout, so it is
// included (not linked) by both .cu translation units; the anonymous namespace gives each
// translation unit its own private copy, which is the normal, ODR-safe pattern for header-only
// CUDA device helpers.

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int64_t kSaiMaxGridDimX = 2147483647;
constexpr int kSaiFastTopKMax = 32;

__device__ __forceinline__ float SaiNegativeInfinity() { return -CUDART_INF_F; }

inline int SaiGridForElements(int64_t count, int threads) {
  const int64_t blocks = (count + threads - 1) / threads;
  return static_cast<int>(std::clamp<int64_t>(blocks, 1, 65535));
}

// ---------------------------------------------------------------------------------------------
// FP32 block reductions
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float SaiBlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

// Reduces (value, index) pairs to the largest value, breaking ties towards the smaller index.
// A negative index marks an empty slot. shared_value/shared_index must already be filled and synced.
__device__ __forceinline__ void SaiBlockArgMax(float* shared_value, int* shared_index) {
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other_index = shared_index[threadIdx.x + stride];
      if (other_index >= 0) {
        const int this_index = shared_index[threadIdx.x];
        const float other_value = shared_value[threadIdx.x + stride];
        const float this_value = shared_value[threadIdx.x];
        if (this_index < 0 || other_value > this_value ||
            (other_value == this_value && other_index < this_index)) {
          shared_value[threadIdx.x] = other_value;
          shared_index[threadIdx.x] = other_index;
        }
      }
    }
    __syncthreads();
  }
}

// Per-thread scan for the best entry that comes strictly after (previous_score, previous_index) in
// the total order "score descending, then index ascending". Entries already emitted are therefore
// skipped without needing a visited bitmap.
__device__ __forceinline__ void SaiScanForNext(const float* scores, int count, float previous_score,
                                               int previous_index, float* best_value, int* best_index) {
  *best_index = -1;
  *best_value = 0.0f;
  for (int candidate = static_cast<int>(threadIdx.x); candidate < count;
       candidate += static_cast<int>(blockDim.x)) {
    const float value = scores[candidate];
    if (previous_index >= 0 &&
        !(value < previous_score || (value == previous_score && candidate > previous_index))) {
      continue;
    }
    if (*best_index < 0 || value > *best_value ||
        (value == *best_value && candidate < *best_index)) {
      *best_value = value;
      *best_index = candidate;
    }
  }
}

__device__ __forceinline__ bool SaiTopKBetter(float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
  return lhs_index >= 0 &&
         (rhs_index < 0 || lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index));
}

// Finds up to kSaiFastTopKMax entries with one global read of each score. Each thread first keeps
// a sorted local list, then adjacent threads merge their lists in shared memory. The final list is
// ordered by score descending and index ascending, matching SaiScanForNext/SaiBlockArgMax.
__device__ __forceinline__ void SaiBlockTopK(const float* scores, int count, int top_k,
                                             float* shared_value, int* shared_index) {
  float local_value[kSaiFastTopKMax];
  int local_index[kSaiFastTopKMax];
  for (int rank = 0; rank < top_k; ++rank) {
    local_value[rank] = 0.0f;
    local_index[rank] = -1;
  }

  for (int candidate = static_cast<int>(threadIdx.x); candidate < count;
       candidate += static_cast<int>(blockDim.x)) {
    const float value = scores[candidate];
    if (!SaiTopKBetter(value, candidate, local_value[top_k - 1], local_index[top_k - 1])) {
      continue;
    }
    int rank = top_k - 1;
    while (rank > 0 && SaiTopKBetter(value, candidate, local_value[rank - 1], local_index[rank - 1])) {
      local_value[rank] = local_value[rank - 1];
      local_index[rank] = local_index[rank - 1];
      --rank;
    }
    local_value[rank] = value;
    local_index[rank] = candidate;
  }

  const int thread_offset = static_cast<int>(threadIdx.x) * top_k;
  for (int rank = 0; rank < top_k; ++rank) {
    shared_value[thread_offset + rank] = local_value[rank];
    shared_index[thread_offset + rank] = local_index[rank];
  }
  __syncthreads();

  for (int stride = 1; stride < static_cast<int>(blockDim.x); stride <<= 1) {
    if ((static_cast<int>(threadIdx.x) % (2 * stride)) == 0) {
      const int right_offset = (static_cast<int>(threadIdx.x) + stride) * top_k;
      for (int rank = 0; rank < top_k; ++rank) {
        local_value[rank] = shared_value[thread_offset + rank];
        local_index[rank] = shared_index[thread_offset + rank];
      }

      int left_rank = 0;
      int right_rank = 0;
      for (int rank = 0; rank < top_k; ++rank) {
        const bool take_right =
            right_rank < top_k &&
            (left_rank == top_k ||
             SaiTopKBetter(shared_value[right_offset + right_rank], shared_index[right_offset + right_rank],
                           local_value[left_rank], local_index[left_rank]));
        if (take_right) {
          shared_value[thread_offset + rank] = shared_value[right_offset + right_rank];
          shared_index[thread_offset + rank] = shared_index[right_offset + right_rank];
          ++right_rank;
        } else {
          shared_value[thread_offset + rank] = local_value[left_rank];
          shared_index[thread_offset + rank] = local_index[left_rank];
          ++left_rank;
        }
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
// Rotary embeddings
// ---------------------------------------------------------------------------------------------

// Split-half rotary over the leading `rotary_width` channels (the convention used by the qsa
// reference). Channels beyond `rotary_width` pass through unchanged.
template <typename T>
__device__ __forceinline__ float SaiLeadingRope(const float* value, int rotary_width, const T* cos_row,
                                                const T* sin_row, int d) {
  if (d >= rotary_width) {
    return value[d];
  }
  const int half = rotary_width / 2;
  const float paired = (d < half) ? -value[d + half] : value[d - half];
  return value[d] * to_float<T>(cos_row[d]) + paired * to_float<T>(sin_row[d]);
}

// Interleaved rotary over the trailing 2 * rotary_width channels (the convention used by the csa
// reference). Each cos/sin entry covers one channel pair, matching repeat_interleave(2).
template <typename T>
__device__ __forceinline__ float SaiTrailingRope(const float* value, int head_size, int rotary_width,
                                                 const T* cos_row, const T* sin_row, int d) {
  const int base = head_size - 2 * rotary_width;
  if (d < base) {
    return value[d];
  }
  const int offset = d - base;
  const float paired = ((offset & 1) == 0) ? -value[d + 1] : value[d - 1];
  return value[d] * to_float<T>(cos_row[offset >> 1]) + paired * to_float<T>(sin_row[offset >> 1]);
}

// ---------------------------------------------------------------------------------------------
// Causal geometry
// ---------------------------------------------------------------------------------------------

// Highest compressed entry a query at `position` may attend to, matching (position + 1) // ratio.
__device__ __forceinline__ int64_t SaiCausalThreshold(int64_t position, int compress_ratio) {
  return position < 0 ? 0 : position / compress_ratio + (position % compress_ratio == compress_ratio - 1);
}

__device__ __forceinline__ int SaiClampPosition(int64_t position, int max_rotary_length) {
  if (position < 0) {
    return 0;
  }
  const int64_t limit = max_rotary_length - 1;
  return static_cast<int>(position < limit ? position : limit);
}

}  // namespace

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
