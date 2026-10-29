// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>

#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Decode routing in one launch: softmax + top-k (bit-identical to SoftmaxTopKMergeKernel<T, 128, 4>) followed by the
// (expert, token)-sorted expert maps of fusedBuildExpertMapsSortFirstTokenKernel.
//
// One warp per row. Lane l emulates the four "threads" v = l + 32 w (w = 0..3) of the 128-thread merge kernel, each of
// which owns logits [4v, 4v + 4): the max / exp-sum reduction trees are reproduced operation for operation
// (cub::BlockReduce<float, 128>: ascending warp shuffle tree per 32 threads, the four warp aggregates summed in order),
// but entirely in registers. The top-k selection is exact on unique keys (score, lower index first), so the selection
// algorithm cannot change the result: logits are fp16 / bf16 values, so the sortable 32-bit score has 13 zero low bits
// and a 32-bit key (score | inverted index) orders exactly like the 64-bit key of the merge kernel.
constexpr int kFusedRouteMaxRows = 8;
constexpr int kFusedRouteMaxK = 16;
constexpr int kFusedRouteExperts = 512;  // the kernel is specialised for exactly 512 experts
constexpr int kFusedRouteThreads = 256;
static_assert(kFusedRouteMaxRows * kFusedRouteMaxK <= 128);

__device__ __forceinline__ uint32_t FusedRouteKey(float logit, int index, bool valid) {
  const uint32_t bits = logit == 0.0f ? 0u : __float_as_uint(logit);
  const uint32_t sortable = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return valid ? ((sortable & 0xFFFFE000u) | (0x1FFFu - static_cast<uint32_t>(index))) : 0u;
}

__device__ __forceinline__ float FusedRouteScore(uint32_t key) {
  const uint32_t sortable = key & 0xFFFFE000u;
  return __uint_as_float((sortable & 0x80000000u) ? (sortable & 0x7FFFFFFFu) : (~sortable & 0xFFFFE000u));
}

__device__ __forceinline__ void FusedRouteLoad4(const half* p, float (&v)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const half2 a = *reinterpret_cast<const half2*>(&raw.x);
  const half2 b = *reinterpret_cast<const half2*>(&raw.y);
  v[0] = __low2float(a);
  v[1] = __high2float(a);
  v[2] = __low2float(b);
  v[3] = __high2float(b);
}

__device__ __forceinline__ void FusedRouteLoad4(const __nv_bfloat16* p, float (&v)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  v[0] = __uint_as_float(raw.x << 16);
  v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
  v[2] = __uint_as_float(raw.y << 16);
  v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
}

// K > 0: top-k count known at compile time (fully unrolled selection); K == 0: runtime k.
template <typename T, int K>
__global__ void __launch_bounds__(kFusedRouteThreads) FusedSoftmaxTopKMapsKernel(
    const T* logits, float* topk_scales, int* topk_indices, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int* permuted_expert, int64_t* expert_first_token_offset, int num_rows,
    int k_runtime, bool normalize_scales) {
  constexpr int num_experts = kFusedRouteExperts;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Programmatic dependent launch: the router logits come from the previous kernel; let the expert GEMV kernel
  // start its prologue right away (it waits for this grid to complete before reading the maps).
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  constexpr int kMaxSel = K > 0 ? K : kFusedRouteMaxK;
  const int k = K > 0 ? K : k_runtime;
  __shared__ int s_key[kFusedRouteMaxRows * kFusedRouteMaxK];
  __shared__ int s_sorted[256];

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  s_sorted[threadIdx.x] = INT_MAX;  // padding for the branchless searches below

  if (warp < num_rows) {
    const T* row_logits = logits + static_cast<size_t>(warp) * num_experts;  // 8-byte aligned (checked by the launcher)
    float scores[16];
    uint32_t keys[16];
    float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int base = (lane + 32 * w) * 4;
      float v[4];
      FusedRouteLoad4(row_logits + base, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        scores[w * 4 + j] = v[j] == 0.0f ? 0.0f : v[j];
        keys[w * 4 + j] = FusedRouteKey(v[j], base + j, true);
        local_max = fmaxf(local_max, v[j]);
      }
    }
    // Max over all experts (order independent: the per-lane values are never NaN).
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
      local_max = (local_max < other) ? other : local_max;
    }
    const float max_val = local_max;

    // k rounds of an exact warp-wide arg-max (independent of the softmax sums below, so the scheduler overlaps them).
    uint32_t winners[kMaxSel];
#pragma unroll
    for (int r = 0; r < kMaxSel; ++r) {
      if (K > 0 || r < k) {
        uint32_t best = keys[0];
#pragma unroll
        for (int j = 1; j < 16; ++j) {
          best = max(best, keys[j]);
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        winners[r] = __reduce_max_sync(0xFFFFFFFFu, best);
#else
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
          best = max(best, __shfl_xor_sync(0xFFFFFFFFu, best, offset));
        }
        winners[r] = best;
#endif
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          keys[j] = keys[j] == winners[r] ? 0u : keys[j];
        }
      } else {
        winners[r] = 0u;
      }
    }

    float part[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum += expf(scores[w * 4 + j] - max_val);
      }
      part[w] = sum;
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        part[w] += __shfl_xor_sync(0xFFFFFFFFu, part[w], offset);
      }
    }
    // lane 0 holds the tree result that cub's shfl_down reduction leaves in lane 0
    const float block_sum = ((__shfl_sync(0xFFFFFFFFu, part[0], 0) + __shfl_sync(0xFFFFFFFFu, part[1], 0)) +
                             __shfl_sync(0xFFFFFFFFu, part[2], 0)) +
                            __shfl_sync(0xFFFFFFFFu, part[3], 0);
    const float inv_sum = onnxruntime::cuda::topk::SafeInvSum(block_sum);

    uint32_t selected = 0;
    float prob = 0.0f;
    float scale_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < kMaxSel; ++r) {
      if (K > 0 || r < k) {
        const float p = onnxruntime::cuda::topk::SoftmaxScale(FusedRouteScore(winners[r]), max_val, inv_sum);
        scale_sum += p;  // sequential rank order, as thread 0 of the merge kernel
        if (lane == r) {
          selected = winners[r];
          prob = p;
        }
      }
    }
    const float denom = onnxruntime::cuda::topk::TopKNormalizeDenom(normalize_scales, scale_sum);
    if (lane < k) {
      const int expert = 0x1FFF - static_cast<int>(selected & 0x1FFFu);
      topk_scales[static_cast<size_t>(warp) * k + lane] = normalize_scales ? (prob / denom) : prob;
      topk_indices[static_cast<size_t>(warp) * k + lane] = expert;
      s_key[warp * k + lane] = (expert << 3) | warp;  // unique per (expert, token); sorts by expert, then token
    }
  }
  const int num_pairs = num_rows * k;
  __syncthreads();

  // Expert maps: pair p = (token p / k, rank p % k) is ranked by (expert, token), the order of the radix rank kernel.
  if (static_cast<int>(threadIdx.x) < num_pairs) {
    const int p = static_cast<int>(threadIdx.x);
    const int key = s_key[p];
    int rank = 0;
    for (int q = 0; q < num_pairs; ++q) {
      rank += s_key[q] < key ? 1 : 0;
    }
    const int token = key & 7;
    const int unpermuted_row = (p - token * k) * num_rows + token;
    const int expert = key >> 3;
    permuted_row_to_unpermuted_row[rank] = unpermuted_row;
    unpermuted_row_to_permuted_row[unpermuted_row] = rank;
    permuted_expert[rank] = expert;
    s_sorted[rank] = expert;
  }
  __syncthreads();
  // expert_first_token_offset[e] = number of pairs routed to an expert < e (e = 0..num_experts)
  int count[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int e = static_cast<int>(threadIdx.x) + i * kFusedRouteThreads;
    count[i] = 0;
#pragma unroll
    for (int step = 128; step >= 1; step >>= 1) {
      count[i] += s_sorted[count[i] + step - 1] < e ? step : 0;
    }
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int e = static_cast<int>(threadIdx.x) + i * kFusedRouteThreads;
    if (e <= num_experts) {
      expert_first_token_offset[e] = count[i];
    }
  }
}

// Fused 1-CTA decode routing + paired-first expert route grouping for INT3 QMoE (1..8 rows, 512 experts, k <= 16).
// Bit-identical to SoftmaxTopKMergeKernel<T, 128, 4>, and simultaneously emits paired-first int2 route_groups.
template <typename T, int K>
__global__ void __launch_bounds__(kFusedRouteThreads) FusedInt3SoftmaxTopKGroupsKernel(
    const T* logits, float* topk_scales, int* topk_indices, int2* route_groups,
    int num_rows, int k_runtime, bool normalize_scales) {
  constexpr int num_experts = kFusedRouteExperts;
  constexpr int kMaxSel = K > 0 ? K : kFusedRouteMaxK;
  const int k = K > 0 ? K : k_runtime;
  __shared__ int s_key[kFusedRouteMaxRows * kFusedRouteMaxK];
  __shared__ int s_sorted_expert[kFusedRouteMaxRows * kFusedRouteMaxK];
  __shared__ int s_sorted_route[kFusedRouteMaxRows * kFusedRouteMaxK];
  __shared__ int s_warp_paired[8];
  __shared__ int s_warp_single[8];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (warp < num_rows) {
    const T* row_logits = logits + static_cast<size_t>(warp) * num_experts;
    float scores[16];
    uint32_t keys[16];
    float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int base = (lane + 32 * w) * 4;
      float v[4];
      FusedRouteLoad4(row_logits + base, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        scores[w * 4 + j] = v[j] == 0.0f ? 0.0f : v[j];
        keys[w * 4 + j] = FusedRouteKey(v[j], base + j, true);
        local_max = fmaxf(local_max, v[j]);
      }
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
      local_max = (local_max < other) ? other : local_max;
    }
    const float max_val = local_max;

    uint32_t winners[kMaxSel];
#pragma unroll
    for (int r = 0; r < kMaxSel; ++r) {
      if (K > 0 || r < k) {
        uint32_t best = keys[0];
#pragma unroll
        for (int j = 1; j < 16; ++j) {
          best = max(best, keys[j]);
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        winners[r] = __reduce_max_sync(0xFFFFFFFFu, best);
#else
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
          best = max(best, __shfl_xor_sync(0xFFFFFFFFu, best, offset));
        }
        winners[r] = best;
#endif
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          keys[j] = keys[j] == winners[r] ? 0u : keys[j];
        }
      } else {
        winners[r] = 0u;
      }
    }

    float part[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum += expf(scores[w * 4 + j] - max_val);
      }
      part[w] = sum;
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        part[w] += __shfl_xor_sync(0xFFFFFFFFu, part[w], offset);
      }
    }
    const float block_sum = ((__shfl_sync(0xFFFFFFFFu, part[0], 0) + __shfl_sync(0xFFFFFFFFu, part[1], 0)) +
                             __shfl_sync(0xFFFFFFFFu, part[2], 0)) +
                            __shfl_sync(0xFFFFFFFFu, part[3], 0);
    const float inv_sum = onnxruntime::cuda::topk::SafeInvSum(block_sum);

    uint32_t selected = 0;
    float prob = 0.0f;
    float scale_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < kMaxSel; ++r) {
      if (K > 0 || r < k) {
        const float p = onnxruntime::cuda::topk::SoftmaxScale(FusedRouteScore(winners[r]), max_val, inv_sum);
        scale_sum += p;
        if (lane == r) {
          selected = winners[r];
          prob = p;
        }
      }
    }
    const float denom = onnxruntime::cuda::topk::TopKNormalizeDenom(normalize_scales, scale_sum);
    if (lane < k) {
      const int expert = 0x1FFF - static_cast<int>(selected & 0x1FFFu);
      const int orig_route = warp * k + lane;
      topk_scales[static_cast<size_t>(orig_route)] = normalize_scales ? (prob / denom) : prob;
      topk_indices[static_cast<size_t>(orig_route)] = expert;
      s_key[orig_route] = (expert << 7) | orig_route;
    }
  }
  const int num_pairs = num_rows * k;
  __syncthreads();

  if (tid < num_pairs) {
    const int my_key = s_key[tid];
    int rank = 0;
    for (int q = 0; q < num_pairs; ++q) {
      rank += (s_key[q] < my_key) ? 1 : 0;
    }
    s_sorted_expert[rank] = my_key >> 7;
    s_sorted_route[rank] = my_key & 0x7F;
  }
  __syncthreads();

  bool is_paired = false;
  bool is_single = false;
  int2 grp = make_int2(-1, -1);
  if (tid < num_pairs) {
    const int e = s_sorted_expert[tid];
    int pos_in_run = 0;
    for (int d = 1; d <= tid && s_sorted_expert[tid - d] == e; ++d) {
      ++pos_in_run;
    }
    if ((pos_in_run & 1) == 0) {
      const int r0 = s_sorted_route[tid];
      const int r1 = (tid + 1 < num_pairs && s_sorted_expert[tid + 1] == e) ? s_sorted_route[tid + 1] : -1;
      grp = make_int2(r0, r1);
      if (r1 >= 0) {
        is_paired = true;
      } else {
        is_single = true;
      }
    }
  }

  const uint32_t paired_mask = __ballot_sync(0xFFFFFFFFu, is_paired);
  const uint32_t single_mask = __ballot_sync(0xFFFFFFFFu, is_single);
  if (lane == 0) {
    s_warp_paired[warp] = __popc(paired_mask);
    s_warp_single[warp] = __popc(single_mask);
  }
  __syncthreads();

  int total_paired = 0;
  int total_single = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    total_paired += s_warp_paired[w];
    total_single += s_warp_single[w];
  }
  const int total_groups = total_paired + total_single;

  if (is_paired) {
    int idx = __popc(paired_mask & ((1u << lane) - 1u));
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) idx += s_warp_paired[w];
    }
    route_groups[idx] = grp;
  } else if (is_single) {
    int idx = total_paired + __popc(single_mask & ((1u << lane) - 1u));
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      if (w < warp) idx += s_warp_single[w];
    }
    route_groups[idx] = grp;
  }
  if (tid >= total_groups && tid < num_pairs) {
    route_groups[tid] = make_int2(-1, -1);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
