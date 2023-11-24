// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>
#include <hip/hip_runtime_api.h>
#include "core/framework/float16.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "contrib_ops/rocm/bert/paged_attention_impl.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

#include "paged_generic.cuh"
#include "paged_dtype_float16.cuh"
#include "paged_dtype_float32.cuh"
#include "paged_utils.cuh"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#include <algorithm>

#define WARP_SIZE 64
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }

  // Broadcast to other threads.
  return __shfl(sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,          // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,        // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ head_mapping,  // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_mapping[head_idx];
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                        + kv_head_idx * kv_head_stride
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                    + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + head_idx * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ head_mapping,  // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {
  paged_attention_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
    /* exp_sums */ nullptr, /* max_logits */ nullptr,
    out, q, k_cache, v_cache, head_mapping, scale, block_tables, context_lens,
    max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,          // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,        // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ tmp_out,        // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ head_mapping,  // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {
  paged_attention_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, head_mapping, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes,
      q_stride, kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int NUM_THREADS,
    int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                          + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = __shfl(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,    // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x + head_idx * (head_size / x) * block_size * x + x_idx * block_size * x + block_offset * x + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + head_offset * block_size + block_offset;
    //{
    //  if (key_cache[tgt_key_idx] - key[src_key_idx] > half(0.1)) {
    //    printf("key error find, %d,%d ", tgt_key_idx, src_key_idx);
    //  }
    //  if (value_cache[tgt_value_idx] - value[src_value_idx] > half(0.1)) {
    //    printf("key error find, %d %d", tgt_value_idx, src_value_idx);
    //  }
    //}
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,       // [batch_size, seq_len] or [num_tokens]
    scalar_t* __restrict__ query,                // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                  // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}

template <typename scalar_t, int repeat, int num_lines_per_thread>
__global__ void repeat_key_value_kernel(
    scalar_t* __restrict__ key_output,    // [num_tokens, repeat*num_head,head_size]
    scalar_t* __restrict__ value_output,  // [num_tokens, repeat*num_head,head_size]
    const scalar_t* __restrict__ key,     // [num_tokens, num_head,head_size]
    const scalar_t* __restrict__ value,   // [num_tokens, num_head,head_size]
    const int head_size, int repeat_def) {
  // const int num_lines_per_thread = 2;
  const int head_idx = blockIdx.x * num_lines_per_thread;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
#pragma unroll
    for (int line = 0; line < num_lines_per_thread; line++) {
      scalar_t b_key = key[(head_idx + line) * head_size + i];
      scalar_t b_value = value[(head_idx + line) * head_size + i];
      if constexpr (repeat > 0) {
#pragma unroll
        for (int j = 0; j < repeat; j++) {
          key_output[((head_idx + line) * repeat + j) * head_size + i] = b_key;
          value_output[((head_idx + line) * repeat + j) * head_size + i] = b_value;
        }
      } else {
#pragma unroll
        for (int j = 0; j < repeat_def; j++) {
          key_output[((head_idx + line) * repeat_def + j) * head_size + i] = b_key;
          value_output[((head_idx + line) * repeat_def + j) * head_size + i] = b_value;
        }
      }
    }
  }
}


}  // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                  \
{                                                                             \
  auto ret = hipFuncSetAttribute(                                               \
      (void*)vllm::paged_attention_v1_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>, \
      hipFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);          \
  ORT_UNUSED_PARAMETER(ret);                                                \
  vllm::paged_attention_v1_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>      \
      <<<grid, block, shared_mem_size, stream>>>(                             \
          out_ptr,                                                            \
          query_ptr,                                                          \
          key_cache_ptr,                                                      \
          value_cache_ptr,                                                    \
          head_mapping_ptr,                                                   \
          scale,                                                              \
          block_tables_ptr,                                                   \
          context_lens_ptr,                                                   \
          max_num_blocks_per_seq,                                             \
          alibi_slopes_ptr,                                                   \
          q_stride,                                                           \
          kv_block_stride,                                                    \
          kv_head_stride);                                                    \
}

// TODO(woosuk): Tune NUM_THREADS.
template <
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    const hipStream_t stream,
    T* out,
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* head_mapping,
    float scale,
    const int* block_tables,
    const int* context_lens,
    int max_context_len,
    const float* alibi_slopes,
    const int64_t max_num_blocks_per_seq,
    const int64_t* query_shapes,
    const int64_t num_queries_per_kv) {
  int num_seqs = query_shapes[0];
  int num_heads = query_shapes[1];
  int head_size = query_shapes[2];
  //int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = num_heads * head_size;  // query.stride(0);
  int kv_block_stride = q_stride / num_queries_per_kv * BLOCK_SIZE;  // key_cache.stride(0);
  int kv_head_stride = head_size * BLOCK_SIZE;//key_cache.stride(1);

#ifndef NDEBUG
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);
#endif

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ? alibi_slopes : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out);
  const T* query_ptr = reinterpret_cast<const T*>(query);
  const T* key_cache_ptr = reinterpret_cast<const T*>(key_cache);
  const T* value_cache_ptr = reinterpret_cast<const T*>(value_cache);
  const int* head_mapping_ptr = reinterpret_cast<const int*>(head_mapping);
  const int* block_tables_ptr = block_tables;
  const int* context_lens_ptr = context_lens;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      // TORCH_CHECK(false, "Unsupported head size: ", head_size);
      abort();
      break;
  }
}

#define CALL_V1_LAUNCHER(T, BLOCK_SIZE)       \
  paged_attention_v1_launcher<T, BLOCK_SIZE>( \
      stream,                                 \
      (T*)out,                                \
      (const T*)query,                        \
      (const T*)key_cache,                    \
      (const T*)value_cache,                  \
      (const int*)head_mapping,               \
      scale,                                  \
      block_tables,                           \
      context_lens,                           \
      max_context_len,                        \
      alibi_slopes,                           \
      max_num_blocks_per_seq,                 \
      query_shapes,                           \
      num_queries_per_kv);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T) \
  switch (block_size) {                \
    case 8:                            \
      CALL_V1_LAUNCHER(T, 8);          \
      break;                           \
    case 16:                           \
      CALL_V1_LAUNCHER(T, 16);         \
      break;                           \
    case 32:                           \
      CALL_V1_LAUNCHER(T, 32);         \
      break;                           \
    default:                           \
      abort();                         \
      break;                           \
  }

void paged_attention_v1(
    const hipStream_t stream,
    void* out,                // [num_seqs, num_heads, head_size]
    const void* query,        // [num_seqs, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* __restrict__ alibi_slopes,
    const int max_num_blocks_per_seq,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype,
    const void* kv_quant_params,  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
    int kv_quant_chunk_size,
    int kv_quant_param_dtype) {
  if (kv_quant_params != nullptr) {
    // TODO: not implemented
    return;
  }
  if (dtype == 0) {  // Float
    CALL_V1_LAUNCHER_BLOCK_SIZE(float);
  } else if (dtype == 1) {  // Half
    CALL_V1_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (dtype == 2) {  // BFloat16
    // CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    // TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                             \
  vllm::paged_attention_v2_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE> \
      <<<grid, block, shared_mem_size, stream>>>(                                        \
          exp_sums_ptr,                                                                  \
          max_logits_ptr,                                                                \
          tmp_out_ptr,                                                                   \
          query_ptr,                                                                     \
          key_cache_ptr,                                                                 \
          value_cache_ptr,                                                               \
          head_mapping_ptr,                                                              \
          scale,                                                                         \
          block_tables_ptr,                                                              \
          context_lens_ptr,                                                              \
          max_num_blocks_per_seq,                                                        \
          alibi_slopes_ptr,                                                              \
          q_stride,                                                                      \
          kv_block_stride,                                                               \
          kv_head_stride);                                                               \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>      \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                          \
          out_ptr,                                                                       \
          exp_sums_ptr,                                                                  \
          max_logits_ptr,                                                                \
          tmp_out_ptr,                                                                   \
          context_lens_ptr,                                                              \
          max_num_partitions);

template <
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 128,
    int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    const hipStream_t stream,
    T* out,
    void* exp_sums,
    void* max_logits,
    void* tmp_out,
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* head_mapping,
    float scale,
    const int* block_tables,
    const int* context_lens,
    int max_context_len,
    const float* alibi_slopes,
    const int64_t max_num_blocks_per_seq,
    const int64_t* query_shapes,
    const int64_t num_queries_per_kv) {
  int num_seqs = query_shapes[0];
  int num_heads = query_shapes[1];
  int head_size = query_shapes[2];
  // int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = num_heads * head_size;                              // query.stride(0);
  int kv_block_stride = q_stride / num_queries_per_kv * BLOCK_SIZE;  // key_cache.stride(0);
  int kv_head_stride = head_size * BLOCK_SIZE;                       // key_cache.stride(1);

#ifndef NDEBUG
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);
#endif

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ? alibi_slopes : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out);
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums);
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits);
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out);

  const T* query_ptr = reinterpret_cast<const T*>(query);
  const T* key_cache_ptr = reinterpret_cast<const T*>(key_cache);
  const T* value_cache_ptr = reinterpret_cast<const T*>(value_cache);
  const int* head_mapping_ptr = reinterpret_cast<const int*>(head_mapping);
  const int* block_tables_ptr = block_tables;
  const int* context_lens_ptr = context_lens;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      abort();
      break;
  }
}

#define CALL_V2_LAUNCHER(T, BLOCK_SIZE)       \
  paged_attention_v2_launcher<T, BLOCK_SIZE>( \
      stream,                                 \
      (T*)out,                                \
      exp_sums,                               \
      max_logits,                             \
      tmp_out,                                \
      (const T*)query,                        \
      (const T*)key_cache,                    \
      (const T*)value_cache,                  \
      (const int*)head_mapping,               \
      scale,                                  \
      block_tables,                           \
      context_lens,                           \
      max_context_len,                        \
      alibi_slopes,                           \
      max_num_blocks_per_seq,                 \
      query_shapes,                           \
      num_queries_per_kv);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T) \
  switch (block_size) {                \
    case 8:                            \
      CALL_V2_LAUNCHER(T, 8);          \
      break;                           \
    case 16:                           \
      CALL_V2_LAUNCHER(T, 16);         \
      break;                           \
    case 32:                           \
      CALL_V2_LAUNCHER(T, 32);         \
      break;                           \
    default:                           \
      abort();                         \
      break;                           \
  }

void paged_attention_v2(
    const hipStream_t stream,
    void* out,                // [num_seqs, num_heads, head_size]
    void* exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    void* max_logits,         // [num_seqs, num_heads, max_num_partitions]
    void* tmp_out,            // [num_seqs, num_heads, max_num_partitions, head_size]
    const void* query,        // [num_seqs, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* alibi_slopes,
    const int max_num_blocks_per_seq,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype) {
  if (dtype == 0) {  // Float
      CALL_V2_LAUNCHER_BLOCK_SIZE(float);
  } else if (dtype == 1) {  // Half
      CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (dtype == 2) {  // BFloat16
    // CALL_V2_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    //TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

void reshape_and_cache(
    const hipStream_t stream,
    const void* key,          // [num_tokens, num_heads, head_size]
    const void* value,        // [num_tokens, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* slot_mapping,  // [num_tokens]
    const int64_t* key_shapes,
    const int64_t* value_shapes,
    const int64_t block_size,
    const int vec_x,
    const int dtype,
    void* kv_quant_param,
    const int kv_quant_chunk_size,
    const int kv_quant_param_dtype) {
  if (kv_quant_param != nullptr) {
    // TODO: not support quantization
    return;
  }
  int num_tokens = key_shapes[0];
  int num_heads = key_shapes[1];
  int head_size = key_shapes[2];
  // int block_size = key_cache.size(3);
  int x = vec_x;

  int key_stride = key_shapes[1] * key_shapes[2];
  int value_stride = value_shapes[1] * value_shapes[2];

  // static_assert(std::is_same_v<T, MLFloat16>, "Unsupported data type: ");

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  // if constexpr (std::is_same_v<T, MLFloat16>) {
  if (dtype == 1) {
    vllm::reshape_and_cache_kernel<half><<<grid, block, 0, stream>>>(
        (const half*)key,
        (const half*)value,
        (half*)key_cache,
        (half*)value_cache,
        slot_mapping,
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
  }
}

void rotary_embedding_neox(
    const hipStream_t stream,
    const int64_t* positions,  // [num_tokens]
    void* query,               // [num_tokens, num_heads * head_size]
    void* key,                 // [num_tokens, num_kv_heads * head_size]
    int head_size,
    const void* cos_sin_cache,  // [max_position, rot_dim]
    int num_tokens,
    int rot_dim,
    int num_heads,
    int num_kv_heads,
    int dtype) {
  // int num_tokens = query.size(0);
  // int rot_dim = cos_sin_cache.size(1);
  // int num_heads = query.size(1) / head_size;
  // int num_kv_heads = key.size(1) / head_size;
  const bool is_neox = true;
  int query_stride = num_heads * head_size;
  int key_stride = num_kv_heads * head_size;
  // TORCH_CHECK(stride == key.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  // const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (dtype == 0) {
    // float
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
    // } else if constexpr (std::is_same_v<T, MLFloat16>) {
  } else if (dtype == 1) {
    // half
    using scalar_t = half;
    if (is_neox) {
      vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions,
          static_cast<scalar_t*>(query),
          static_cast<scalar_t*>(key),
          static_cast<const scalar_t*>(cos_sin_cache),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
    } else {
      vllm::rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          positions,
          static_cast<scalar_t*>(query),
          static_cast<scalar_t*>(key),
          static_cast<const scalar_t*>(cos_sin_cache),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
    }
    //} else if constexpr (std::is_same_v<T, BFloat16>) {
  } else if (dtype == 2) {
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  }
}
template <typename scalar_t>
void LaunchRepeatKeyValue(
    const hipStream_t stream,
    scalar_t* key_out,      // [num_tokens, repeat*num_heads * head_size]
    scalar_t* value_out,    // [num_tokens, repeat*num_heads * head_size]
    const scalar_t* key,    // [num_tokens, num_heads * head_size]
    const scalar_t* value,  // [num_tokens, num_heads * head_size]
    const int64_t* input_shape,
    int repeat) {
  const int unroll_len = 2;
  dim3 grid(input_shape[0] * input_shape[1] / unroll_len);
  dim3 block(input_shape[2] > 256 ? 256 : input_shape[2]);
  if (repeat == 8) {
    vllm::repeat_key_value_kernel<scalar_t, 8, unroll_len><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(key_out),
        static_cast<scalar_t*>(value_out),
        static_cast<const scalar_t*>(key),
        static_cast<const scalar_t*>(value),
        input_shape[2], 0);
  }else if (repeat == 4) {
    vllm::repeat_key_value_kernel<scalar_t, 4, unroll_len><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(key_out),
        static_cast<scalar_t*>(value_out),
        static_cast<const scalar_t*>(key),
        static_cast<const scalar_t*>(value),
        input_shape[2], 0);
  } else {
    vllm::repeat_key_value_kernel<scalar_t, 0, unroll_len><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(key_out),
        static_cast<scalar_t*>(value_out),
        static_cast<const scalar_t*>(key),
        static_cast<const scalar_t*>(value),
        input_shape[2], repeat);
  }
}
template void LaunchRepeatKeyValue<half>(
    const hipStream_t stream,
    half* key_out,      // [num_tokens, repeat*num_heads * head_size]
    half* value_out,    // [num_tokens, repeat*num_heads * head_size]
    const half* key,    // [num_tokens, num_heads * head_size]
    const half* value,  // [num_tokens, num_heads * head_size]
    const int64_t* input_shape,
    int repeat);
template void LaunchRepeatKeyValue<float>(
    const hipStream_t stream,
    float* key_out,      // [num_tokens, repeat*num_heads * head_size]
    float* value_out,    // [num_tokens, repeat*num_heads * head_size]
    const float* key,    // [num_tokens, num_heads * head_size]
    const float* value,  // [num_tokens, num_heads * head_size]
    const int64_t* input_shape,
    int repeat);
#undef DIVIDE_ROUND_UP
#undef WARP_SIZE
#undef MAX
#undef MIN

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
