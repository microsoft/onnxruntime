// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <type_traits>
#include "core/framework/float16.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_common.h"

#include "paged_generic.cuh"
#include "paged_dtype_float16.cuh"
#include "paged_dtype_float32.cuh"
// #include "paged_dtype_bfloat16.cuh"
#include "paged_utils.cuh"

using namespace onnxruntime::cuda;
#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
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
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs).
template <
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
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
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_mapping[head_idx];
  const int seq_idx = blockIdx.y;
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

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
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
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
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
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j <= V_VEC_SIZE; j++) {
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
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
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
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
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
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

template <typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
    const int64_t* __restrict__ positions,       // [num_tokens]
    scalar_t* __restrict__ query,                // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                  // [num_tokens, num_kv_heads, head_size]
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
  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;

    const int rot_offset = i % embed_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;

    const int out_x = token_idx * query_stride + head_idx * head_size + x_index;
    const int out_y = token_idx * query_stride + head_idx * head_size + y_index;

    const scalar_t cos = __ldg(cache_ptr + x_index);
    const scalar_t sin = __ldg(cache_ptr + y_index);

    const scalar_t q_x = query[token_head + x_index];
    const scalar_t q_y = query[token_head + y_index];
    query[out_x] = q_x * cos - q_y * sin;
    query[out_y] = q_y * cos + q_x * sin;
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;

    const int rot_offset = i % embed_dim;
    const int x_index = rot_offset;
    const int y_index = embed_dim + rot_offset;

    const int out_x = token_idx * key_stride + head_idx * head_size + x_index;
    const int out_y = token_idx * key_stride + head_idx * head_size + y_index;

    const scalar_t cos = __ldg(cache_ptr + x_index);
    const scalar_t sin = __ldg(cache_ptr + y_index);

    const scalar_t k_x = key[token_head + x_index];
    const scalar_t k_y = key[token_head + y_index];
    key[out_x] = k_x * cos - k_y * sin;
    key[out_y] = k_y * cos + k_x * sin;
  }
}

}  // namespace vllm

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                 \
  vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS> \
      <<<grid, block, shared_mem_size, stream>>>(                                      \
          out_ptr,                                                                     \
          query_ptr,                                                                   \
          key_cache_ptr,                                                               \
          value_cache_ptr,                                                             \
          head_mapping_ptr,                                                            \
          scale,                                                                       \
          block_tables_ptr,                                                            \
          context_lens_ptr,                                                            \
          max_num_blocks_per_seq,                                                      \
          alibi_slopes_ptr,                                                            \
          query_stride,                                                                \
          kv_block_stride,                                                             \
          kv_head_stride);

// TODO(woosuk): Tune NUM_THREADS.
template <
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
    const cudaStream_t stream,
    T* out,
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int* head_mapping,
    float scale,
    const int* block_tables,
    const int max_num_blocks_per_seq,
    const int* context_lens,
    int max_context_len,
    const float* alibi_slopes_ptr,
    const int64_t* query_shapes,
    int num_queries_per_kv) {
  int num_seqs = query_shapes[0];
  int num_heads = query_shapes[1];
  int head_size = query_shapes[2];
  // int max_num_blocks_per_seq = 1;            // block_tables.size(1);xxxxxxxxxxxxxxxxxxxxxxxx
  int query_stride = head_size * num_heads;                                       // query.stride(0);
  int kv_block_stride = num_heads * head_size / num_queries_per_kv * BLOCK_SIZE;  // xxxxxxxxxxxxxxxxxxxxxxxx
  int kv_head_stride = head_size * BLOCK_SIZE;                                    // key_cache.stride(1);

  // int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % (MAX(WARP_SIZE / BLOCK_SIZE, 1)) == 0);

  T* out_ptr = reinterpret_cast<T*>(out);
  const T* query_ptr = reinterpret_cast<const T*>(query);
  const T* key_cache_ptr = reinterpret_cast<const T*>(key_cache);
  const T* value_cache_ptr = reinterpret_cast<const T*>(value_cache);
  const int* head_mapping_ptr = reinterpret_cast<const int*>(head_mapping);

  const int* block_tables_ptr = block_tables;
  const int* context_lens_ptr = context_lens;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
    // 32, 160, 192, 256.
    // case 32:
    //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
    //   break;
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
      break;
    case 80:
      LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
      break;
    case 96:
      LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
      break;
    // case 160:
    //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
    //   break;
    // case 192:
    //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
    //   break;
    // case 256:
    //   LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
    //   break;
    default:
      // TORCH_CHECK(false, "Unsupported head size: ", head_size);
      abort();
      break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                 \
  single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>( \
      stream,                                               \
      (T*)out,                                              \
      (const T*)query,                                      \
      (const T*)key_cache,                                  \
      (const T*)value_cache,                                \
      (const int*)head_mapping,                             \
      scale,                                                \
      block_tables,                                         \
      max_num_blocks_per_seq,                               \
      context_lens,                                         \
      max_context_len,                                      \
      alibi_slopes_ptr,                                     \
      query_shapes,                                         \
      num_queries_per_kv);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T) \
  switch (block_size) {                    \
    /* case 1:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 1);   */  \
    /*   break;                        */  \
    /* case 2:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 2);   */  \
    /*   break;                        */  \
    /* case 4:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 4);   */  \
    /*   break;                        */  \
    case 8:                                \
      CALL_KERNEL_LAUNCHER(T, 8);          \
      break;                               \
    case 16:                               \
      CALL_KERNEL_LAUNCHER(T, 16);         \
      break;                               \
    case 32:                               \
      CALL_KERNEL_LAUNCHER(T, 32);         \
      break;                               \
    /* case 64:                        */  \
    /*   CALL_KERNEL_LAUNCHER(T, 64);  */  \
    /*   break;                        */  \
    /* case 128:                       */  \
    /*   CALL_KERNEL_LAUNCHER(T, 128); */  \
    /*   break;                        */  \
    /* case 256:                       */  \
    /*   CALL_KERNEL_LAUNCHER(T, 256); */  \
    /*   break;                        */  \
    default:                               \
      abort();                             \
      break;                               \
  }

void single_query_cached_kv_attention(
    const cudaStream_t stream,
    const void* out,           // [num_seqs, num_heads, head_size]
    const void* query,         // [num_seqs, num_heads, head_size]
    const void* key_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const void* value_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    const void* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int max_num_blocks_per_seq,
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* __restrict__ alibi_slopes_ptr,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype) {
  // static_assert(std::is_same_v<T, float> || std::is_same_v<T, BFloat16> || std::is_same_v<T, MLFloat16>, "Unsupported data type: ");
  // if constexpr (std::is_same_v<T, float>) {
  if (dtype == 0) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
    //} else if constexpr (std::is_same_v<T, MLFloat16>) {
  } else if (dtype == 1) {
    // half
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
    //} else if constexpr (std::is_same_v<T, BFloat16>) {
  } else if (dtype == 2) {
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  }
}

void reshape_and_cache(
    const cudaStream_t stream,
    const void* key,          // [num_tokens, num_heads, head_size]
    const void* value,        // [num_tokens, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* slot_mapping,  // [num_tokens]
    const int64_t* key_shapes,
    const int64_t* value_shapes,
    const int64_t block_size,
    const int vec_x,
    int dtype) {
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
    const cudaStream_t stream,
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
  int query_stride = num_heads * head_size;
  int key_stride = num_kv_heads * head_size;
  // TORCH_CHECK(stride == key.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (dtype == 0) {
    // float
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
    // } else if constexpr (std::is_same_v<T, MLFloat16>) {
  } else if (dtype == 1) {
    // half
    using scalar_t = half;
    vllm::rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
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
    //} else if constexpr (std::is_same_v<T, BFloat16>) {
  } else if (dtype == 2) {
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
