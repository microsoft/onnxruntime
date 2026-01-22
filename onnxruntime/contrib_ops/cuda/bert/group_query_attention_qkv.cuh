// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/rotary_common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Fused kernel: Unpack QKV + Apply RoPE to Q and K + Append K/V directly to cache
//
// OPTIMIZATION: This version uses Shared Memory to store the current head being processed.
// Shared memory allows RoPE dispatcher to access paired elements in non-interleaved mode
// (element i pairs with i ± rotary_dim/2) without global memory gathers.
//
// Alignment Note: This kernel assumes that base pointers (packed_qkv, query, etc.)
// are 16-byte aligned and that head_size is a multiple of elements_per_thread.
//
// Grid Layout:
//   blockIdx.x: sequence index (s) -> Max 2^31-1 (Supports very long context)
//   blockIdx.y: head index (head_idx) -> Max 65535
//   blockIdx.z: batch index (b) -> Max 65535
template <typename T, int MAX_HEAD_SIZE = 256>
__global__ void UnpackRoPEAppend(
    const T* packed_qkv,
    const T* query,
    const T* key,
    const T* value,
    T* unpacked_q,
    T* k_cache,
    T* v_cache,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int d,           // packed QKV hidden stride = (num_heads + 2*kv_num_heads) * head_size
    const int max_seqlen,  // KV cache max sequence length
    const int* past_seq_lens,
    const T* cos_cache,
    const T* sin_cache,
    const int rotary_dim,
    const int64_t* position_ids,
    const bool interleaved,
    const bool is_cache_bnsh) {
  using LoadT = float4;
  constexpr int elements_per_thread = sizeof(LoadT) / sizeof(T);

  const int s = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int h = tid * elements_per_thread;

  // Guard work with 'valid' instead of early return to ensure all threads reach __syncthreads()
  const bool valid = (h < head_size);

  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  const int sequence_length = gridDim.x;

  __shared__ T shared_head[MAX_HEAD_SIZE];

  // Determine Head Type and Offset within hidden dimension
  enum HeadType { QUERY,
                  KEY,
                  VALUE };
  HeadType head_type;
  int n;  // Index within its specific type
  int offset_in_hidden;

  if (head_idx < num_heads) {
    head_type = QUERY;
    n = head_idx;
    offset_in_hidden = n * head_size;
  } else if (head_idx < num_heads + kv_num_heads) {
    head_type = KEY;
    n = head_idx - num_heads;
    offset_in_hidden = q_hidden + n * head_size;
  } else {
    head_type = VALUE;
    n = head_idx - (num_heads + kv_num_heads);
    offset_in_hidden = q_hidden + k_hidden + n * head_size;
  }

  // 1. Load data into Registers
  T vals[elements_per_thread];
  if (valid) {
    if (packed_qkv != nullptr) {
      const int64_t packed_idx = static_cast<int64_t>(b) * sequence_length * d +
                                 static_cast<int64_t>(s) * d +
                                 static_cast<int64_t>(offset_in_hidden) + h;
      *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(packed_qkv)[packed_idx / elements_per_thread];
    } else {
      if (head_type == QUERY) {
        const int64_t q_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                              static_cast<int64_t>(s) * q_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(query)[q_idx / elements_per_thread];
      } else if (head_type == KEY) {
        const int64_t k_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                              static_cast<int64_t>(s) * k_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(key)[k_idx / elements_per_thread];
      } else {
        const int64_t v_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                              static_cast<int64_t>(s) * k_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(value)[v_idx / elements_per_thread];
      }
    }
  }

  // 2. Process RoPE
  // Optimization: Only use shared memory for non-interleaved mode
  const bool is_qk = (head_type == QUERY || head_type == KEY);
  if (valid && rotary_dim > 0 && is_qk && !interleaved) {
    T* shared_ptr = &shared_head[h];
    *reinterpret_cast<LoadT*>(shared_ptr) = *reinterpret_cast<LoadT*>(vals);
  }

  // CRITICAL: Barrier must be outside the 'if(valid)' and 'if(is_qk)' blocks
  // to ensure every thread in the block participates.
  __syncthreads();

  if (valid && rotary_dim > 0 && is_qk) {
    const int past_seq_len = past_seq_lens[b];
    const int64_t pos_base = static_cast<int64_t>(b) * sequence_length;
    int pos_id = (position_ids != nullptr) ? static_cast<int>(position_ids[pos_base + s]) : (past_seq_len + s);
    const int h_idx = h / elements_per_thread;

    onnxruntime::contrib::cuda::RotaryDispatcher<LoadT, T>::apply(
        *reinterpret_cast<LoadT*>(vals),
        reinterpret_cast<const LoadT*>(cos_cache),
        reinterpret_cast<const LoadT*>(sin_cache),
        rotary_dim, h_idx, pos_id, interleaved,
        reinterpret_cast<const LoadT*>(shared_head),
        0);
  }

  // 3. Store results back to Global Memory
  if (valid) {
    if (head_type == QUERY) {
      if (unpacked_q != nullptr) {
        const int64_t q_out_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                                  static_cast<int64_t>(s) * q_hidden +
                                  static_cast<int64_t>(n) * head_size + h;
        reinterpret_cast<LoadT*>(unpacked_q)[q_out_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
      }
    } else {
      const int cache_s = past_seq_lens[b] + s;
      if (cache_s < max_seqlen) {
        T* cache_ptr = (head_type == KEY) ? k_cache : v_cache;
        if (cache_ptr != nullptr) {
          int64_t cache_idx = is_cache_bnsh ? (static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size + static_cast<int64_t>(n) * max_seqlen * head_size + static_cast<int64_t>(cache_s) * head_size + h) : (static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size + static_cast<int64_t>(cache_s) * kv_num_heads * head_size + static_cast<int64_t>(n) * head_size + h);
          reinterpret_cast<LoadT*>(cache_ptr)[cache_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
        }
      }
    }
  }
}

template <typename T>
Status LaunchUnpackRoPEAppendKV(
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, T* k_cache, T* v_cache,
    const int num_heads, const int kv_num_heads, const int head_size,
    const int sequence_length, const int batch_size, const int max_seqlen,
    const int* past_seq_lens, const T* cos_cache, const T* sin_cache,
    const int rotary_dim, const int64_t* position_ids, const bool interleaved,
    const bool is_cache_bnsh, cudaStream_t stream, const int max_threads_per_block) {
  constexpr int elements_per_vector = sizeof(float4) / sizeof(T);

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by vector size (16 bytes).");
  }

  // rotary_dim <= head_size check to prevent out-of-bounds in shared memory
  if (rotary_dim > head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "rotary_dim (", rotary_dim, ") cannot exceed head_size (", head_size, ").");
  }

  if (!interleaved && rotary_dim % 2 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Non-interleaved RoPE requires even rotary_dim.");
  }

  const int total_heads = num_heads + 2 * kv_num_heads;
  const int d = total_heads * head_size;

  const int threads_per_block = (head_size + elements_per_vector - 1) / elements_per_vector;
  if (threads_per_block > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size too large for current block configuration.");
  }

  if (total_heads > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Total heads (", total_heads, ") exceeds CUDA grid limit (65535).");
  }
  if (batch_size > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "batch_size (", batch_size, ") exceeds CUDA grid limit (65535).");
  }

  const dim3 grid(sequence_length, total_heads, batch_size);
  const dim3 block(threads_per_block);

  // Dynamic dispatch for MAX_HEAD_SIZE templates to improve occupancy for common LLM head sizes
  if (head_size <= 64) {
    UnpackRoPEAppend<T, 64><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh);
  } else if (head_size <= 128) {
    UnpackRoPEAppend<T, 128><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh);
  } else if (head_size <= 256) {
    UnpackRoPEAppend<T, 256><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size (", head_size, ") exceeds maximum supported MAX_HEAD_SIZE (256).");
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status LaunchUnpackRoPEAppendKV<half>(
    const half*, const half*, const half*, const half*, half*, half*, half*,
    int, int, int, int, int, int, const int*, const half*, const half*, int, const int64_t*, bool, bool,
    cudaStream_t, int);

template Status LaunchUnpackRoPEAppendKV<BFloat16>(
    const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, BFloat16*, BFloat16*, BFloat16*,
    int, int, int, int, int, int, const int*, const BFloat16*, const BFloat16*, int, const int64_t*, bool, bool,
    cudaStream_t, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
