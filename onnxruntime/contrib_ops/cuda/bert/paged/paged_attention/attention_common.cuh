// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/tensor.hpp"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"

namespace onnxruntime::contrib::paged {

// Utility function for attention softmax.
template <int NUM_WARPS>
__forceinline__ __device__ float
block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / constant::WarpSize;
  int lane = threadIdx.x % constant::WarpSize;

  CUTE_UNROLL
  for (int mask = constant::WarpSize / 2; mask >= 1; mask /= 2) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  if (lane == 0) {
    red_smem[warp] = sum;
  }

  __syncthreads();

  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  CUTE_UNROLL
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return SHFL_SYNC(sum, 0);
}

template <int NumThreads, int GroupSize, int NumWarps>
__forceinline__ __device__ void
softmax_cta(
    float* reduction_buffer, float qk_max_of_group,
    float* logits, int logits_size,      // buffer desc
    int start_tok_idx, int end_tok_idx,  // valid range
    float2* max_sum = nullptr            // optional output for max and sum of the chunk
) {
  auto lane_idx = lane_id();
  float qk_max = qk_max_of_group;
  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The leading thread of each group already has its max qk value.
  CUTE_UNROLL
  for (int mask = constant::WarpSize / 2; mask >= GroupSize; mask /= 2) {
    qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane_idx == 0) {
    reduction_buffer[warp_id()] = qk_max;
  }
  __syncthreads();

  // Get the max qk value for the sequence.
  qk_max = lane_idx < NumWarps ? reduction_buffer[lane_idx] : std::numeric_limits<float>::lowest();
  CUTE_UNROLL
  for (int mask = NumWarps / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = threadIdx.x; i < logits_size; i += NumThreads) {
    float val = (start_tok_idx <= i && i < end_tok_idx) ? __expf(logits[i] - qk_max) : 0.0f;
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NumWarps>(&reduction_buffer[NumWarps], exp_sum);

  // Compute softmax.
  const float inv_sum = __frcp_rn(exp_sum + 1e-6f);
  for (int i = threadIdx.x; i < logits_size; i += NumThreads) {
    logits[i] *= inv_sum;
  }

  if (max_sum != nullptr && threadIdx.x == 0) {
    *max_sum = float2{qk_max, exp_sum};
  }
  __syncthreads();
}

}  // namespace onnxruntime::contrib::paged
