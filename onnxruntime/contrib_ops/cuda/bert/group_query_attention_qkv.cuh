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

// Fused kernel: Unpack QKV + Apply RoPE to Q and K + Append K/V directly to cache + Quantize if needed
//
// This kernel performs the following:
// 1. Unpacks Q, K, V from input tensor(s). The input can be a single packed QKV tensor
//    or three separate Q, K, V tensors.
// 2. Applies Rotary Positional Embedding (RoPE) to Q and K if rotary_dim > 0.
// 3. Appends K and V to the KV cache at the correct sequence index (past_seq_len + s).
//    - Performs on-the-fly quantization (Int8 or Int4) if configured (BIT_WIDTH < 16).
//    - Supports both BNSH and BSNH layouts for the KV cache.
// 4. Writes the rotated Q back to global memory (unpacked_q) for the subsequent attention kernel.
//
// Template Parameters:
// - T: The floating point type (half or BFloat16).
// - BIT_WIDTH: The bit width for KV cache quantization (16=none, 8=Int8, 4=Int4).
// - MAX_HEAD_SIZE: Maximum supported head size, used for shared memory allocation.
template <typename T, int BIT_WIDTH = 16, int MAX_HEAD_SIZE = 256>
__global__ void UnpackRoPEAppend(
    const T* packed_qkv,
    const T* query,
    const T* key,
    const T* value,
    T* unpacked_q,
    void* k_cache,
    void* v_cache,
    const float* k_scale,
    const float* v_scale,
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
    const bool is_cache_bnsh,
    const bool per_channel) {
  using LoadT = float4;
  constexpr int elements_per_thread = sizeof(LoadT) / sizeof(T);

  // Determine grid coordinates:
  // - s: current sequence index (within the new tokens batch)
  // - head_idx: global head index (0 to num_heads + 2*kv_num_heads - 1)
  // - b: batch index
  const int s = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  // h: the starting channel index for this thread (multiple elements per thread via LoadT)
  const int h = tid * elements_per_thread;

  // Guard work with 'valid' instead of early return to ensure all threads reach __syncthreads()
  const bool valid = (h < head_size);

  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  const int sequence_length = gridDim.x;  // Number of new tokens in this launch

  __shared__ T shared_head[MAX_HEAD_SIZE];

  // Determine Head Type and Offset within the packed hidden dimension [Q, K, V]
  enum HeadType { QUERY,
                  KEY,
                  VALUE };
  HeadType head_type;
  int n;  // Index relative to its specific type
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
  alignas(16) T vals[elements_per_thread];
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

  // 2. Process RoPE (Rotary Positional Embedding)
  // Non-interleaved RoPE requires full head visibility to pair channels (h, h + d/2).
  // We use shared memory as a staging buffer to allow any thread to access its pair.
  const bool is_qk = (head_type == QUERY || head_type == KEY);
  if (valid && rotary_dim > 0 && is_qk && !interleaved) {
    T* shared_ptr = &shared_head[h];
    *reinterpret_cast<LoadT*>(shared_ptr) = *reinterpret_cast<LoadT*>(vals);
  }

  // CRITICAL: Barrier must be outside the 'if(valid)' and 'if(is_qk)' blocks
  // to ensure every thread in the block participates and shared memory is ready.
  __syncthreads();

  if (valid && rotary_dim > 0 && is_qk) {
    const int past_seq_len = past_seq_lens[b];
    const int64_t pos_base = static_cast<int64_t>(b) * sequence_length;
    // Calculate global position for RoPE: use position_ids if provided, else rely on past_seq_len.
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

  // 3. Store results back to Global Memory (Unpacked Q and Quantized KV Cache)
  if (valid) {
    if (head_type == QUERY) {
      if (unpacked_q != nullptr) {
        // Store rotated Q to global memory for the Attention kernel
        const int64_t q_out_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                                  static_cast<int64_t>(s) * q_hidden +
                                  static_cast<int64_t>(n) * head_size + h;
        reinterpret_cast<LoadT*>(unpacked_q)[q_out_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
      }
    } else {
      // Store K or V into the KV cache at index (past_seqlen + s)
      const int cache_s = past_seq_lens[b] + s;
      if (cache_s < max_seqlen) {
        void* cache_ptr = (head_type == KEY) ? k_cache : v_cache;
        if (cache_ptr != nullptr) {
          int64_t cache_idx;
          if (is_cache_bnsh) {
            // BNSH layout: [Batch, NumHeads, SeqLen, HeadSize]
            // Note: For BIT_WIDTH=4, head_size refers to the number of UNPACKED elements.
            // stride_s is the number of bytes occupied by head_size elements.
            const int64_t stride_s = (BIT_WIDTH == 4) ? (head_size / 2) : head_size;
            const int64_t stride_n = max_seqlen * stride_s;
            const int64_t stride_b = kv_num_heads * stride_n;
            cache_idx = static_cast<int64_t>(b) * stride_b +
                        static_cast<int64_t>(n) * stride_n +
                        static_cast<int64_t>(cache_s) * stride_s +
                        (BIT_WIDTH == 4 ? h / 2 : h);
          } else {
            // BSNH layout: [Batch, SeqLen, NumHeads, HeadSize]
            const int64_t stride_n = (BIT_WIDTH == 4) ? (head_size / 2) : head_size;
            const int64_t stride_s = kv_num_heads * stride_n;
            const int64_t stride_b = max_seqlen * stride_s;
            cache_idx = static_cast<int64_t>(b) * stride_b +
                        static_cast<int64_t>(cache_s) * stride_s +
                        static_cast<int64_t>(n) * stride_n +
                        (BIT_WIDTH == 4 ? h / 2 : h);
          }

          if constexpr (BIT_WIDTH == 16 || BIT_WIDTH == 32) {
            // No quantization: direct store
            reinterpret_cast<LoadT*>(cache_ptr)[cache_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
          } else if constexpr (BIT_WIDTH == 8) {
            // Int8 Quantization: 1 element per byte
            const float* scale_buffer = (head_type == KEY) ? k_scale : v_scale;
            uint64_t packed = 0;
            for (int i = 0; i < elements_per_thread; ++i) {
              float sc = per_channel ? scale_buffer[n * head_size + h + i] : scale_buffer[0];
              float inv_s = (sc == 0.0f) ? 0.0f : 1.0f / sc;
              int8_t q = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(static_cast<float>(vals[i]) * inv_s))));
              packed |= (static_cast<uint64_t>(static_cast<uint8_t>(q)) << (i * 8));
            }
            // Store 8 elements (8 bytes) at once
            reinterpret_cast<uint64_t*>(cache_ptr)[cache_idx / 8] = packed;
          } else if constexpr (BIT_WIDTH == 4) {
            // Int4 Quantization: 2 elements per byte
            constexpr float kInt4Min = -8.0f;
            constexpr float kInt4Max = 7.0f;
            const float* scale_buffer = (head_type == KEY) ? k_scale : v_scale;
            uint32_t packed = 0;
            for (int i = 0; i < 4; ++i) {
              // Elements are paired as (0,1), (2,3), etc. into single bytes.
              float s0 = per_channel ? scale_buffer[n * head_size + h + i * 2] : scale_buffer[0];
              float s1 = per_channel ? scale_buffer[n * head_size + h + i * 2 + 1] : scale_buffer[0];
              int8_t q0 = static_cast<int8_t>(max(kInt4Min, min(kInt4Max, rintf(static_cast<float>(vals[i * 2]) * (s0 == 0 ? 0 : 1.0f / s0)))));
              int8_t q1 = static_cast<int8_t>(max(kInt4Min, min(kInt4Max, rintf(static_cast<float>(vals[i * 2 + 1]) * (s1 == 0 ? 0 : 1.0f / s1)))));
              uint8_t p = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
              packed |= (static_cast<uint32_t>(p) << (i * 8));
            }
            // Store 8 elements (4 bytes) at once
            reinterpret_cast<uint32_t*>(cache_ptr)[cache_idx / 4] = packed;
          }
        }
      }
    }
  }
}

// Internal dispatcher that selects the appropriate template specialization based on head_size.
// MAX_HEAD_SIZE is used to optimize shared memory usage and kernel performance.
template <typename T, int BIT_WIDTH>
Status DispatchUnpackRoPEAppendHeadSize(
    const dim3& grid, const dim3& block, cudaStream_t stream,
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, void* k_cache, void* v_cache,
    const float* k_scale, const float* v_scale,
    const int num_heads, const int kv_num_heads, const int head_size, const int d,
    const int max_seqlen, const int* past_seq_lens,
    const T* cos_cache, const T* sin_cache, const int rotary_dim,
    const int64_t* position_ids, const bool interleaved, const bool is_cache_bnsh, const bool per_channel) {
  if (head_size <= 64) {
    UnpackRoPEAppend<T, BIT_WIDTH, 64><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (head_size <= 128) {
    UnpackRoPEAppend<T, BIT_WIDTH, 128><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (head_size <= 256) {
    UnpackRoPEAppend<T, BIT_WIDTH, 256><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size (", head_size, ") exceeds maximum supported MAX_HEAD_SIZE (256).");
  }
  return CUDA_CALL(cudaGetLastError());
}

// Public entry point to launch the Unpack+RoPE+Append kernel.
// Handles parameter validation, grid/block sizing, and bit-width dispatching.
template <typename T>
Status LaunchUnpackRoPEAppend(
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, void* k_cache, void* v_cache,
    const float* k_scale, const float* v_scale,
    const int num_heads, const int kv_num_heads, const int head_size,
    const int sequence_length, const int batch_size, const int max_seqlen,
    const int* past_seq_lens, const T* cos_cache, const T* sin_cache,
    const int rotary_dim, const int64_t* position_ids, const bool interleaved,
    const bool is_cache_bnsh, const KVQuantizationType k_quant_type,
    const int bit_width, cudaStream_t stream, const int max_threads_per_block) {
  constexpr int elements_per_vector = sizeof(float4) / sizeof(T);

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by vector size (16 bytes).");
  }

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

  bool per_channel = (k_quant_type == KVQuantizationType::PER_CHANNEL);

  if (bit_width == 0) {
    return DispatchUnpackRoPEAppendHeadSize<T, 16>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (bit_width == 8) {
    return DispatchUnpackRoPEAppendHeadSize<T, 8>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
#ifdef USE_INT4_KV_CACHE
  } else if (bit_width == 4) {
    return DispatchUnpackRoPEAppendHeadSize<T, 4>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
#endif
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported bit_width (", bit_width, ") for GQA quantization.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
