// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include <cublas_v2.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

////////// Auxiliary Kernels

template <typename T>
__global__ void UnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                    const int kv_num_heads, const int head_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= token_count * (num_heads + 2 * kv_num_heads) * head_size) {
    return;
  }
  const int q_hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int in_seq_stride = q_hidden_size + 2 * kv_hidden_size;

  int packed_i;
  if (tid < token_count * q_hidden_size) {
    const int token_id = tid / q_hidden_size;
    const int offset = tid % q_hidden_size;
    packed_i = token_id * in_seq_stride + offset;
  } else if (tid < token_count * (q_hidden_size + kv_hidden_size)) {
    const int id = tid - token_count * q_hidden_size;
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + offset;
  } else if (tid < token_count * (q_hidden_size + 2 * kv_hidden_size)) {
    const int id = tid - token_count * (q_hidden_size + kv_hidden_size);
    const int token_id = id / kv_hidden_size;
    const int offset = id % kv_hidden_size;
    packed_i = token_id * in_seq_stride + q_hidden_size + kv_hidden_size + offset;
  }
  unpacked_qkv[tid] = packed_qkv[packed_i];
}

// Since QKV is unpacked into a single workspace buffer, this is similar to a transpose
template <typename T>
Status LaunchUnpackQKVCumulative(const T* packed_qkv, T* unpacked_qkv, const int token_count, const int num_heads,
                                 const int kv_num_heads, const int head_size, cudaStream_t stream,
                                 const int max_threads_per_block) {
  const int threads = max_threads_per_block;
  const int blocks = (token_count * (num_heads + 2 * kv_num_heads) * head_size + threads - 1) / threads;
  UnpackQKVCumulative<T><<<blocks, threads, 0, stream>>>(packed_qkv, unpacked_qkv, token_count, num_heads, kv_num_heads,
                                                         head_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void UnpackV(const T* input, T* output, const int token_count, const int hidden_size,
                        const int packed_seq_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < token_count * hidden_size) {
    int offset = tid % hidden_size;
    int token_id = tid / hidden_size;
    int packed_i = token_id * packed_seq_stride + offset;
    output[tid] = input[packed_i];
  }
}

template <typename T>
Status LaunchUnpackCumulative(const T* input, T* output, const int token_count, const int hidden_size,
                              const int packed_seq_stride, cudaStream_t stream, const int max_threads_per_block) {
  const int threads = std::min(max_threads_per_block, token_count * hidden_size);
  const int blocks = (token_count * hidden_size + threads - 1) / threads;
  UnpackV<T><<<blocks, threads, 0, stream>>>(input, output, token_count, hidden_size, packed_seq_stride);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void RotaryEmbeddingTNH(T* output,                            // TxNxH
                                   const T* input,                       // TxNxH
                                   const T* cos_cache,                   // Mx(H/2)
                                   const T* sin_cache,                   // Mx(H/2)
                                   const int32_t* past_seqlens,          // B
                                   const int32_t* cumulative_seqlens_q,  // B+1
                                   const int head_size,
                                   const int rotary_embedding_dim,
                                   const bool interleaved,
                                   const int3 in_strides,     // TxNxH
                                   const int3 out_strides) {  // TxNxH
  // Use .x in innermost loop to access global memory efficiently

  const int b = blockIdx.y;
  const int s = blockIdx.x;
  const int n = blockIdx.z;
  const int h = threadIdx.x;

  const int sequence_length = cumulative_seqlens_q[b + 1] - cumulative_seqlens_q[b];
  if (h >= head_size || s >= sequence_length) {
    return;
  }

  const int t = cumulative_seqlens_q[b] + s;  // t is the index of the token in the unpadded input/output
  const T* input_data = input + t * in_strides.x + n * in_strides.y;
  T* output_data = output + t * out_strides.x + n * out_strides.y;

  if (h >= rotary_embedding_dim) {
    output_data[h] = input_data[h];
    return;
  }

  // Cache is (M, H/2)
  const int half_rotary_embedding_dim = rotary_embedding_dim / 2;
  const int position_id = past_seqlens[b] + s;
  const int cache_offset = position_id * half_rotary_embedding_dim;
  const T* cos_data = cos_cache + cache_offset;
  const T* sin_data = sin_cache + cache_offset;

  int cache_idx = 0;
  T sign = 0;
  int j = 0;
  if (interleaved) {
    cache_idx = (h / 2) % half_rotary_embedding_dim;
    sign = (h % 2 == 0) ? -1 : 1;
    j = (h % 2 == 0) ? h + 1 : h - 1;  // i - sign
  } else {
    cache_idx = h % half_rotary_embedding_dim;
    sign = (h < half_rotary_embedding_dim) ? -1 : 1;
    j = (h + half_rotary_embedding_dim) % rotary_embedding_dim;
  }
  output_data[h] = input_data[h] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
}

template <typename T>
Status LaunchRotaryEmbeddingKernel(cudaStream_t stream, T* output, const T* input, const int32_t* past_seqlens,
                                   const int32_t* cumulative_seqlens_q, const T* cos_cache, const T* sin_cache,
                                   const int batch_size, const int max_seqlen_q, const int num_heads,
                                   const int head_size, const int rotary_embedding_dim, const bool interleaved,
                                   const int in_seq_stride, const int max_threads_per_block) {
  ORT_ENFORCE(head_size <= max_threads_per_block, "Rotary embedding dim must be <= max_threads_per_block");
  int3 in_strides = {in_seq_stride <= 0 ? num_heads * head_size : in_seq_stride, head_size, 1};
  int3 out_strides = {num_heads * head_size, head_size, 1};
  int tpb = (head_size + 31) / 32 * 32;

  const dim3 grid(max_seqlen_q, batch_size, num_heads);
  const dim3 block(tpb);
  RotaryEmbeddingTNH<<<grid, block, 0, stream>>>(
      output, input, cos_cache, sin_cache, past_seqlens, cumulative_seqlens_q, head_size, rotary_embedding_dim,
      interleaved, in_strides, out_strides);
  return CUDA_CALL(cudaGetLastError());
}

template <int kBlockSize>
__global__ void GetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                       const int32_t* past_seqlens, const int batch_size) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id == 0) {
    cumulative_seqlens_kv[0] = 0;
  }

  typedef cub::BlockScan<int, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Sum past_seqlens to new sequence length (which we get by subtracting cumulative_seqlens_q).
  // Then do an inclusive sum across present sequence lengths to get the cumulative sequence length
  if (id < batch_size) {
    cumulative_seqlens_kv[id + 1] = past_seqlens[id] + cumulative_seqlens_q[id + 1] - cumulative_seqlens_q[id];
    int length = cumulative_seqlens_kv[id + 1];
    BlockScan(temp_storage).InclusiveSum(length, length);
    cumulative_seqlens_kv[id + 1] = length;
  }
}

Status LaunchGetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                    const int32_t* past_seqlens, const int batch_size, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  GetCumulativeSeqlensKV<256><<<blocks, threads, 0, stream>>>(cumulative_seqlens_kv, cumulative_seqlens_q, past_seqlens,
                                                              batch_size);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
__global__ void ReshapeAndCache(const T* __restrict__ key, const T* __restrict__ value, T* __restrict__ key_cache,
                                T* __restrict__ value_cache, const int* __restrict__ block_table,
                                const int* __restrict__ past_seqlens, const int* __restrict__ cumulative_seqlens_q,
                                const int batch_size, const int max_num_blocks_per_seq, const int token_count,
                                const int kv_hidden_size, const int block_size, const int key_stride,
                                const int value_stride) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= token_count * kv_hidden_size) {
    return;
  }
  const int token_id = tid / kv_hidden_size;
  const int hidden_offset = tid % kv_hidden_size;
  int batch_id = 0;
  for (int i = 0; i < batch_size; ++i) {
    if (token_id < cumulative_seqlens_q[i + 1]) {
      batch_id = i;
      break;
    }
  }
  const int token_offset = token_id - cumulative_seqlens_q[batch_id];
  const int past_length = past_seqlens[batch_id];
  const int block_id = block_table[batch_id * max_num_blocks_per_seq + (past_length + token_offset) / block_size];
  const int block_offset = (past_length + token_offset) % block_size;

  const int key_id = token_id * key_stride + hidden_offset;
  const int value_id = token_id * value_stride + hidden_offset;
  const int dst_id = block_id * block_size * kv_hidden_size + block_offset * kv_hidden_size + hidden_offset;
  key_cache[dst_id] = key[key_id];
  value_cache[dst_id] = value[value_id];
}

template <typename T>
Status LaunchReshapeAndCache(const T* key, const T* value, T* key_cache, T* value_cache, const int* block_table,
                             const int* past_seqlens, const int* cumulative_seqlens_q, const int batch_size,
                             const int max_num_blocks_per_seq, const int token_count, const int kv_hidden_size,
                             const int block_size, const int key_stride, const int value_stride, cudaStream_t stream,
                             const int max_threads_per_block) {
  const int total_size = token_count * kv_hidden_size;
  const int threads(std::min(total_size, max_threads_per_block));
  const int blocks((total_size + threads - 1) / threads);
  ReshapeAndCache<T><<<blocks, threads, 0, stream>>>(key, value, key_cache, value_cache, block_table, past_seqlens,
                                                     cumulative_seqlens_q, batch_size, max_num_blocks_per_seq,
                                                     token_count, kv_hidden_size, block_size, key_stride, value_stride);
  return CUDA_CALL(cudaGetLastError());
}

// Gather paged KV into packed-varlen [total_kv_tokens, num_heads, head_size], expanding GQA heads.
// total_elems = total_kv_tokens * num_heads * head_size can exceed INT32_MAX for realistic
// large-context GQA configs (e.g., 2M tokens * 64 * 128 = 16.4B), so the linear index is int64_t
// and the kernel uses a grid-stride loop instead of a single (tid >= total_elems) early-exit.
template <typename T>
__global__ void GatherAndExpandPagedKVCache(const T* __restrict__ key_cache,
                                            const T* __restrict__ value_cache,
                                            T* __restrict__ gathered_key,
                                            T* __restrict__ gathered_value,
                                            const int* __restrict__ block_table,
                                            const int* __restrict__ cumulative_seqlens_kv,
                                            const int batch_size,
                                            const int num_heads,
                                            const int kv_num_heads,
                                            const int head_size,
                                            const int block_size,
                                            const int max_num_blocks_per_seq,
                                            const int64_t total_elems) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t num_heads_times_head = static_cast<int64_t>(num_heads) * head_size;
  const int q_kv_head_ratio = num_heads / kv_num_heads;
  const int64_t page_stride = static_cast<int64_t>(block_size) * kv_num_heads * head_size;

  for (int64_t tid = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
       tid < total_elems;
       tid += stride) {
    const int h = static_cast<int>(tid % head_size);
    const int head_id = static_cast<int>((tid / head_size) % num_heads);
    const int token_id = static_cast<int>(tid / num_heads_times_head);

    // cumulative_seqlens_kv is a prefix sum of non-negative per-batch KV lengths
    // (past_seqlens[i] + new_tokens[i]), so it is monotonically non-decreasing for
    // any valid op input — the same assumption the previous linear scan made.
    // Binary-search for the batch this token belongs to: log2(batch_size) is strictly
    // better than the linear scan, which ran once per (token, head, h) element and
    // multiplied its cost by num_heads * head_size.
    int left = 0;
    int right = batch_size;
    while (left < right) {
      const int mid = left + (right - left) / 2;
      if (token_id < cumulative_seqlens_kv[mid + 1]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    const int batch_id = left;

    const int pos = token_id - cumulative_seqlens_kv[batch_id];
    const int block_idx_in_seq = pos / block_size;
    const int block_offset = pos % block_size;
    const int block_id = block_table[batch_id * max_num_blocks_per_seq + block_idx_in_seq];

    // GQA expansion: each output head maps to kv_head_id = head_id / (num_heads / kv_num_heads).
    // For MHA (num_heads == kv_num_heads) this is the identity.
    const int kv_head_id = head_id / q_kv_head_ratio;

    const int64_t paged_idx = static_cast<int64_t>(block_id) * page_stride +
                              static_cast<int64_t>(block_offset) * kv_num_heads * head_size +
                              kv_head_id * head_size +
                              h;

    gathered_key[tid] = key_cache[paged_idx];
    gathered_value[tid] = value_cache[paged_idx];
  }
}

template <typename T>
Status LaunchGatherAndExpandPagedKVCache(const T* key_cache, const T* value_cache,
                                         T* gathered_key, T* gathered_value,
                                         const int* block_table, const int* cumulative_seqlens_kv,
                                         const int batch_size, const int num_heads,
                                         const int kv_num_heads, const int head_size,
                                         const int block_size, const int max_num_blocks_per_seq,
                                         const int total_kv_tokens, cudaStream_t stream,
                                         const int max_threads_per_block) {
  const int64_t total_elems = static_cast<int64_t>(total_kv_tokens) * num_heads * head_size;
  if (total_elems == 0) {
    return Status::OK();
  }
  // With the op's batch_size <= 256 precondition (paged_attention.cc) and MEA's
  // head_size <= 1024 cap, blocks_needed = ceil(total_elems / threads) stays comfortably
  // within int range for any realistic input, so no explicit clamp is needed. The kernel
  // uses a grid-stride loop so launching fewer blocks than total_elems / threads would
  // also be correct — we don't need an artificial "keep SMs busy" cap.
  const int threads = static_cast<int>(std::min<int64_t>(max_threads_per_block, total_elems));
  const int blocks = static_cast<int>((total_elems + threads - 1) / threads);
  GatherAndExpandPagedKVCache<T><<<blocks, threads, 0, stream>>>(
      key_cache, value_cache, gathered_key, gathered_value,
      block_table, cumulative_seqlens_kv,
      batch_size, num_heads, kv_num_heads, head_size,
      block_size, max_num_blocks_per_seq, total_elems);
  return CUDA_CALL(cudaGetLastError());
}

////////// Launch Kernels

#if USE_FLASH_ATTENTION
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T>& data,
    float scale) {
  // Get parameters
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const float softcap = parameters.softcap;
  bool is_bf16 = std::is_same<T, BFloat16>::value;
  const int local_window_size = parameters.local_window_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int block_size = parameters.block_size;
  // The following are passed to flash api but not used by the kernel, so they can be determined heuristically
  const int max_query_len = token_count - batch_size + 1;
  const int max_seq_len = parameters.max_num_blocks_per_seq * parameters.block_size;

  T* query = const_cast<T*>(data.query);
  T* key;
  T* value;
  if (!parameters.is_packed_qkv) {
    key = const_cast<T*>(data.key);
    value = const_cast<T*>(data.value);
  } else {
    key = reinterpret_cast<T*>(query) + static_cast<size_t>(num_heads * head_size);
    value = reinterpret_cast<T*>(key) + static_cast<size_t>(kv_num_heads * head_size);
  }

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across FA and MEA dispatch paths so the host can also read total_kv_tokens.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;

  if (parameters.do_rotary) {
    // Will unpack Q and K in case of packed_qkv
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, kv_num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
  } else if (parameters.is_packed_qkv) {
    // Only unpack Q. K and V are unpacked by ReshapeAndCache.
    auto q_buffer = data.workspace_buffer;
    const int packed_seq_stride = q_hidden_size + 2 * kv_hidden_size;
    ORT_RETURN_IF_ERROR(LaunchUnpackCumulative<T>(
        query, q_buffer, token_count, q_hidden_size, packed_seq_stride, stream, max_threads_per_block));
    query = q_buffer;
  }

  // Insert key and value into block-based KV cache
  int* block_table = const_cast<int*>(data.block_table);
  const int key_stride = parameters.is_packed_qkv && !parameters.do_rotary ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  ORT_RETURN_IF_ERROR(LaunchReshapeAndCache<T>(key, value, data.key_cache, data.value_cache, block_table, past_seqlens,
                                               cumulative_seqlens_q, batch_size, max_num_blocks_per_seq, token_count,
                                               kv_hidden_size, block_size, key_stride, value_stride, stream,
                                               max_threads_per_block));

  // Launch kernel
  void* q = reinterpret_cast<void*>(query);
  void* key_cache = reinterpret_cast<void*>(data.key_cache);
  void* value_cache = reinterpret_cast<void*>(data.value_cache);
  void* output = reinterpret_cast<void*>(data.output);
  void* softmax_lse = reinterpret_cast<void*>(data.softmax_lse);
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_varlen_fwd(
      device_prop, stream, q, key_cache, value_cache, output, cumulative_seqlens_q, cumulative_seqlens_kv,
      /*seqused_k*/ nullptr, block_table, softmax_lse, batch_size, num_heads, kv_num_heads, head_size, max_query_len,
      max_seq_len, token_count, scale, softcap, /*is_causal*/ true, is_bf16, local_window_size - 1, max_num_blocks_per_seq,
      block_size));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("flash attention output", data.output, token_count, num_heads, head_size);

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
// Fallback when FlashAttention is unavailable (SM<80 or ORT_DISABLE_FLASH_ATTENTION=1).
// Mirrors the FlashAttention preprocessing (rotary, unpack, ReshapeAndCache), then gathers
// the paged KV cache into a packed-varlen [total_kv_tokens, num_heads, head_size] buffer and
// dispatches to CUTLASS memory-efficient attention via its seqstart_q / seqstart_k varlen ABI.
// Caller must populate data.gathered_key / data.gathered_value / data.total_kv_tokens.
template <typename T>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T>& data,
    float scale) {
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int token_count = parameters.token_count;
  const int q_hidden_size = parameters.hidden_size;
  const int kv_hidden_size = parameters.kv_hidden_size;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const int block_size = parameters.block_size;
  const int max_num_blocks_per_seq = parameters.max_num_blocks_per_seq;
  const int local_window_size = parameters.local_window_size;
  const int total_kv_tokens = data.total_kv_tokens;
  // Use the caller-computed actual max of per-batch new-query lengths, not the
  // `token_count - batch_size + 1` heuristic: the heuristic assumes >=1 new token per batch
  // and underestimates otherwise, which would silently drop query tokens from the
  // rotary grid and from MEA's `grid_x = ceil_div(sequence_length, kQueriesPerBlock)`.
  const int max_query_len = data.max_query_len;

  T* query = const_cast<T*>(data.query);
  T* key;
  T* value;
  if (!parameters.is_packed_qkv) {
    key = const_cast<T*>(data.key);
    value = const_cast<T*>(data.value);
  } else {
    key = reinterpret_cast<T*>(query) + static_cast<size_t>(num_heads * head_size);
    value = reinterpret_cast<T*>(key) + static_cast<size_t>(kv_num_heads * head_size);
  }

  // cumulative_seqlens_kv is populated by the caller (paged_attention.cc) before QkvToContext;
  // shared across FA and MEA dispatch paths.
  int* cumulative_seqlens_q = const_cast<int*>(data.cumulative_seqlens_q);
  int* past_seqlens = const_cast<int*>(data.past_seqlens);
  int* cumulative_seqlens_kv = data.cumulative_seqlens_kv;

  if (parameters.do_rotary) {
    auto q_buffer = data.workspace_buffer;
    auto k_buffer = data.workspace_buffer + token_count * num_heads * head_size;
    const int packed_seq_stride = parameters.is_packed_qkv ? (num_heads + 2 * kv_num_heads) * head_size : -1;
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, q_buffer, query, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(
        stream, k_buffer, key, past_seqlens, cumulative_seqlens_q, data.cos_cache, data.sin_cache, batch_size,
        max_query_len, kv_num_heads, head_size, parameters.rotary_dim, parameters.rotary_interleaved, packed_seq_stride,
        max_threads_per_block));
    query = q_buffer;
    key = k_buffer;
  } else if (parameters.is_packed_qkv) {
    auto q_buffer = data.workspace_buffer;
    const int packed_seq_stride = q_hidden_size + 2 * kv_hidden_size;
    ORT_RETURN_IF_ERROR(LaunchUnpackCumulative<T>(
        query, q_buffer, token_count, q_hidden_size, packed_seq_stride, stream, max_threads_per_block));
    query = q_buffer;
  }

  int* block_table = const_cast<int*>(data.block_table);
  const int key_stride = parameters.is_packed_qkv && !parameters.do_rotary ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  const int value_stride = parameters.is_packed_qkv ? q_hidden_size + 2 * kv_hidden_size : kv_hidden_size;
  ORT_RETURN_IF_ERROR(LaunchReshapeAndCache<T>(key, value, data.key_cache, data.value_cache, block_table, past_seqlens,
                                               cumulative_seqlens_q, batch_size, max_num_blocks_per_seq, token_count,
                                               kv_hidden_size, block_size, key_stride, value_stride, stream,
                                               max_threads_per_block));

  ORT_RETURN_IF_ERROR(LaunchGatherAndExpandPagedKVCache<T>(
      data.key_cache, data.value_cache, data.gathered_key, data.gathered_value,
      block_table, cumulative_seqlens_kv, batch_size, num_heads, kv_num_heads,
      head_size, block_size, max_num_blocks_per_seq, total_kv_tokens, stream, max_threads_per_block));

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_bf16 = std::is_same<T, BFloat16>::value;
  p.is_half = !p.is_bf16 && (sizeof(T) == 2);
  p.batch_size = batch_size;
  p.num_heads = num_heads;
  p.sequence_length = max_query_len;
  p.kv_sequence_length = total_kv_tokens;
  p.max_sequence_length = total_kv_tokens;
  p.qk_head_size = head_size;
  p.v_head_size = head_size;
  p.causal = true;
  p.scale = scale;
  p.softcap = parameters.softcap;
  p.local_window_size = local_window_size;
  p.seqstart_q_ptr = cumulative_seqlens_q;
  p.seqstart_k_ptr = cumulative_seqlens_kv;
  p.seqlen_k_ptr = nullptr;
  p.query = query;
  p.key = data.gathered_key;
  p.value = data.gathered_value;
  p.attn_bias = nullptr;
  p.is_kv_bsnh = true;
  p.has_custom_right_padding = false;
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(head_size, sizeof(T) == sizeof(float))
                    ? data.fmha_buffer
                    : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("mea paged attention output", data.output, token_count, num_heads, head_size);

  return Status::OK();
}
#endif

////////// API Functions

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& /*cublas*/,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size)) : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data, scale);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return EfficientAttention(device_prop, stream, parameters, data, scale);
  }
#endif

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No PagedAttention kernel available for the current configuration.");
}

template struct PagedAttentionData<half>;
template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<half>& data);

template struct PagedAttentionData<BFloat16>;
template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<BFloat16>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
