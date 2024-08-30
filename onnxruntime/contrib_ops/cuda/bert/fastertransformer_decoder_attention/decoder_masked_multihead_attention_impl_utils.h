/*
 * The implementation of this file is based on code provided by https://github.com/NVIDIA/FasterTransformer
 *
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modifications Copyright (c) Microsoft.
// Licensed under the MIT License.

// Modifications:
// (1) Minor routine name changes for integration into the ORT code-base

#pragma once

#include "contrib_ops/cuda/bert/utils.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace decoder_masked_self_attention_details {

//------------------------------------------------------------
// Block_sum
//------------------------------------------------------------

template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

//------------------------------------------------------------
// Shfl_Mask
//------------------------------------------------------------

inline __device__ constexpr uint32_t shfl_mask(int threads) {
  return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

//------------------------------------------------------------
// Dot
//------------------------------------------------------------

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

//------------------------------------------------------------
// Qk_Dot
//------------------------------------------------------------

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N]) {
  using K_vec_acum = K_vec;

  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = onnxruntime::cuda::fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

template <typename T, int THREADS_PER_KEY>
struct Qk_dot {
  template <typename K_vec, int N>
  static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N]) {
    return qk_dot_<THREADS_PER_KEY>(q, k);
  }
};

//------------------------------------------------------------
// ThreadsPerValue
//------------------------------------------------------------

template <typename T, int head_size>
struct ThreadsPerValue {
  static const int value = head_size * sizeof(T) / 16;
};

//------------------------------------------------------------
// CalcDynamicBlockMemory
//------------------------------------------------------------

template <typename T>
inline size_t CalcDynamicBlockMemory(const DecoderMaskedMultiHeadAttentionParams& params,
                                     int threads_per_value, int threads_per_block) {
  // The amount of shared memory needed to store the Q*K^T values in float.

  const int total_sequence_length = params.total_sequence_length;
  size_t qk_sz = (((total_sequence_length + 3) / 4) * 16);

  // The extra memory needed if we are not using floats for the final logits.
  size_t logits_sz = 0;

  if (sizeof(T) != 4) {
    logits_sz = (((total_sequence_length + 3) / 4) * 4 * sizeof(T));
  }

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;

  // The amount of storage needed to finalize the outputs.
  size_t red_sz = rows_per_red * params.head_size * sizeof(T) / 2;

  size_t transpose_rotary_size = 0;
  if (params.rotary_embedding_dim > 0) {
    transpose_rotary_size = 2 * params.rotary_embedding_dim * sizeof(T);
  }

  // The max.
  return std::max(std::max(softmax_sz, red_sz), transpose_rotary_size);
}

}  // namespace decoder_masked_self_attention_details
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
