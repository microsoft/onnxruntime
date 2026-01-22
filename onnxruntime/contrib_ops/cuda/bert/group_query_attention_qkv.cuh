// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Enable quantized KV cache support for INT8/INT4
#define KV_QUANT_SUPPORTED 0

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
// This kernel performs the following operations in a fused manner:
// 1. Unpacks packed QKV (if packed) or reads separated Q, K, V
// 2. Applies Rotary Positional Embedding (RoPE) to Q and K
// 3. Appends K and V to the KV cache (past_key/past_value)
// 4. Quantizes K and V if T_QUANT is different from T (and KV_QUANT_SUPPORTED is enabled)
//
// Template Parameters:
//   T         - Input/Output data type (e.g., half, nv_bfloat16)
//   T_QUANT   - Quantized data type for KV cache (e.g., int8_t, uint4_t packed in wider type)
//   T_SCALE   - Scale data type for quantization (e.g., float)
//   bit_width - Bit width for quantization (8 or 4)
//
// Arguments:
//   packed_qkv    - Input packed QKV tensor [B, S, (H_q + 2*H_v)*D] (Optional)
//   query         - Input Q tensor (if not packed)
//   key           - Input K tensor (if not packed)
//   value         - Input V tensor (if not packed)
//   unpacked_q    - Output buffer for rotary encodded query [B, S, H_q, D]
//   k_cache       - Output KV cache for key
//   v_cache       - Output KV cache for value
//   k_scale       - Scale factor for key quantization
//   v_scale       - Scale factor for value quantization
//   num_heads     - Number of query heads (H_q)
//   kv_num_heads  - Number of KV heads (H_v)
//   head_size     - Dimension of each head (D)
//   d             - Stride for packed QKV hidden dimension
//   max_seqlen    - Maximum sequence length of the KV cache
//   past_seq_lens - Sequence lengths of past tokens (where to append new tokens)
//   cos_cache     - RoPE cosine table
//   sin_cache     - RoPE sine table
//   rotary_dim    - Dimension to apply RoPE
//   position_ids  - Position indices for RoPE
//   interleaved   - Whether RoPE uses interleaved (x, x+D/2) or adjacent (x, x+1) pairs
//   is_cache_bnsh - Layout of KV cache (true: BNSH, false: BSNH)
//   per_channel   - Quantization granularity (true: per-channel, false: per-tensor)
template <typename T, typename T_QUANT, typename T_SCALE, int bit_width>
__global__ void UnpackRoPEQuantizeAppend(
    const T* packed_qkv,
    const T* query,
    const T* key,
    const T* value,
    T* unpacked_q,
    T_QUANT* k_cache,
    T_QUANT* v_cache,
    const T_SCALE* k_scale,
    const T_SCALE* v_scale,
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
  // Vectorized load/store using float4 (16 bytes)
  using LoadT = float4;
  constexpr int elements_per_thread = sizeof(LoadT) / sizeof(T);

  const int b = blockIdx.z;
  const int s = blockIdx.y;
  const int offset_vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int offset = offset_vec_idx * elements_per_thread;

  if (offset >= d) return;

  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  const int sequence_length = gridDim.y;

  // 1. Load data into registers
  // NOTE: When packed_qkv is nullptr, separate Q/K/V inputs MUST be in BSNH layout
  // (batch, sequence, num_heads, head_size). The indexing below assumes this format.
  T vals[elements_per_thread];
  if (packed_qkv != nullptr) {
    const int64_t packed_idx = static_cast<int64_t>(b) * sequence_length * d +
                               static_cast<int64_t>(s) * d + offset;
    *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(packed_qkv)[packed_idx / elements_per_thread];
  } else {
    // Separate Q/K/V inputs - assumes BSNH layout
    if (offset < q_hidden) {
      const int64_t q_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                            static_cast<int64_t>(s) * q_hidden + offset;
      *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(query)[q_idx / elements_per_thread];
    } else if (offset < q_hidden + k_hidden) {
      const int64_t k_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                            static_cast<int64_t>(s) * k_hidden + (offset - q_hidden);
      *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(key)[k_idx / elements_per_thread];
    } else {
      const int64_t v_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                            static_cast<int64_t>(s) * k_hidden + (offset - q_hidden - k_hidden);
      *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(value)[v_idx / elements_per_thread];
    }
  }

  // Common RoPE Calculations
  const int past_seq_len = past_seq_lens[b];
  int pos_id = (position_ids != nullptr) ? static_cast<int>(position_ids[b * sequence_length + s]) : (past_seq_len + s);

  // 2. Process based on component (Q, K, or V)
  if (offset < q_hidden) {
    // --- QUERY ---
    if (cos_cache != nullptr && rotary_dim > 0) {
      const int q_head_idx = offset / head_size;
      const int h = offset % head_size;
      const int h_idx = h / elements_per_thread;

      const T* src_for_rope = (packed_qkv != nullptr) ? packed_qkv : query;
      int64_t src_offset_for_rope = (packed_qkv != nullptr) ? (static_cast<int64_t>(b) * sequence_length * d + static_cast<int64_t>(s) * d + static_cast<int64_t>(q_head_idx) * head_size) : (static_cast<int64_t>(b) * sequence_length * q_hidden + static_cast<int64_t>(s) * q_hidden + static_cast<int64_t>(q_head_idx) * head_size);

      RotaryDispatcher<LoadT, T>::apply(
          *reinterpret_cast<LoadT*>(vals),
          reinterpret_cast<const LoadT*>(cos_cache),
          reinterpret_cast<const LoadT*>(sin_cache),
          rotary_dim, h_idx, pos_id, interleaved,
          reinterpret_cast<const LoadT*>(src_for_rope),
          src_offset_for_rope / elements_per_thread);
    }
    if (unpacked_q != nullptr) {
      const int64_t q_out_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                                static_cast<int64_t>(s) * q_hidden + offset;
      reinterpret_cast<LoadT*>(unpacked_q)[q_out_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
    }

  } else if (offset < q_hidden + k_hidden) {
    // --- KEY ---
    const int k_offset = offset - q_hidden;
    const int n = k_offset / head_size;
    const int h = k_offset % head_size;
    const int h_idx = h / elements_per_thread;

    if (cos_cache != nullptr && rotary_dim > 0) {
      const T* src_for_rope = (packed_qkv != nullptr) ? packed_qkv : key;
      int64_t src_offset_for_rope = (packed_qkv != nullptr) ? (static_cast<int64_t>(b) * sequence_length * d + static_cast<int64_t>(s) * d + q_hidden + static_cast<int64_t>(n) * head_size) : (static_cast<int64_t>(b) * sequence_length * k_hidden + static_cast<int64_t>(s) * k_hidden + static_cast<int64_t>(n) * head_size);

      RotaryDispatcher<LoadT, T>::apply(
          *reinterpret_cast<LoadT*>(vals),
          reinterpret_cast<const LoadT*>(cos_cache),
          reinterpret_cast<const LoadT*>(sin_cache),
          rotary_dim, h_idx, pos_id, interleaved,
          reinterpret_cast<const LoadT*>(src_for_rope),
          src_offset_for_rope / elements_per_thread);
    }

    const int cache_s = past_seq_len + s;
    int64_t cache_idx_base = is_cache_bnsh ? (static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size + static_cast<int64_t>(n) * max_seqlen * head_size + static_cast<int64_t>(cache_s) * head_size) : (static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size + static_cast<int64_t>(cache_s) * kv_num_heads * head_size + static_cast<int64_t>(n) * head_size);

    // Quantize and Store K
    if (k_cache != nullptr && cache_s < max_seqlen) {
      if constexpr (sizeof(T_QUANT) == sizeof(T)) {
        reinterpret_cast<LoadT*>(k_cache)[(cache_idx_base + h) / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
      } else {
#if KV_QUANT_SUPPORTED
        if constexpr (bit_width == 8) {
          uint64_t packed = 0;
          for (int i = 0; i < elements_per_thread; ++i) {
            float scale_val = per_channel ? static_cast<float>(k_scale[n * head_size + h + i]) : static_cast<float>(k_scale[0]);
            float inv_s = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
            int8_t q = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(static_cast<float>(vals[i]) * inv_s))));
            packed |= (static_cast<uint64_t>(static_cast<uint8_t>(q)) << (i * 8));
          }
          reinterpret_cast<uint64_t*>(k_cache)[(cache_idx_base + h) / elements_per_thread] = packed;
        } else if constexpr (bit_width == 4) {
          uint32_t packed = 0;
          for (int i = 0; i < 4; ++i) {  // 4 bytes, each having 2 values
            float scale0 = per_channel ? static_cast<float>(k_scale[n * head_size + h + i * 2]) : static_cast<float>(k_scale[0]);
            float scale1 = per_channel ? static_cast<float>(k_scale[n * head_size + h + i * 2 + 1]) : static_cast<float>(k_scale[0]);
            int8_t q0 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(static_cast<float>(vals[i * 2]) * (scale0 == 0 ? 0 : 1.0f / scale0)))));
            int8_t q1 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(static_cast<float>(vals[i * 2 + 1]) * (scale1 == 0 ? 0 : 1.0f / scale1)))));
            uint8_t p = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
            packed |= (static_cast<uint32_t>(p) << (i * 8));
          }
          reinterpret_cast<uint32_t*>(k_cache)[(cache_idx_base + h) / elements_per_thread] = packed;
        }
#endif
      }
    }

  } else {
    // --- VALUE ---
    const int v_offset = offset - q_hidden - k_hidden;
    const int n = v_offset / head_size;
    const int h = v_offset % head_size;

    const int cache_s = past_seq_len + s;
    int64_t cache_idx_base = is_cache_bnsh ? (static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size + static_cast<int64_t>(n) * max_seqlen * head_size + static_cast<int64_t>(cache_s) * head_size) : (static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size + static_cast<int64_t>(cache_s) * kv_num_heads * head_size + static_cast<int64_t>(n) * head_size);

    // Quantize and Store V
    if (v_cache != nullptr && cache_s < max_seqlen) {
      if constexpr (sizeof(T_QUANT) == sizeof(T)) {
        reinterpret_cast<LoadT*>(v_cache)[(cache_idx_base + h) / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
      } else {
#if KV_QUANT_SUPPORTED
        if constexpr (bit_width == 8) {
          uint64_t packed = 0;
          for (int i = 0; i < elements_per_thread; ++i) {
            float scale_val = per_channel ? static_cast<float>(v_scale[n * head_size + h + i]) : static_cast<float>(v_scale[0]);
            float inv_s = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
            int8_t q = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(static_cast<float>(vals[i]) * inv_s))));
            packed |= (static_cast<uint64_t>(static_cast<uint8_t>(q)) << (i * 8));
          }
          reinterpret_cast<uint64_t*>(v_cache)[(cache_idx_base + h) / elements_per_thread] = packed;
        } else if constexpr (bit_width == 4) {
          uint32_t packed = 0;
          for (int i = 0; i < 4; ++i) {
            float scale0 = per_channel ? static_cast<float>(v_scale[n * head_size + h + i * 2]) : static_cast<float>(v_scale[0]);
            float scale1 = per_channel ? static_cast<float>(v_scale[n * head_size + h + i * 2 + 1]) : static_cast<float>(v_scale[0]);
            int8_t q0 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(static_cast<float>(vals[i * 2]) * (scale0 == 0 ? 0 : 1.0f / scale0)))));
            int8_t q1 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(static_cast<float>(vals[i * 2 + 1]) * (scale1 == 0 ? 0 : 1.0f / scale1)))));
            uint8_t p = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
            packed |= (static_cast<uint32_t>(p) << (i * 8));
          }
          reinterpret_cast<uint32_t*>(v_cache)[(cache_idx_base + h) / elements_per_thread] = packed;
        }
#endif
      }
    }
  }
}

// Dispatcher for different bit-widths of the fused kernel
template <typename T, typename T_QUANT, typename T_SCALE, int bit_width>
void DispatchUnpackRoPEQuantizeAppend(
    const dim3& grid, const dim3& block, cudaStream_t stream,
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, T_QUANT* k_cache, T_QUANT* v_cache,
    const T_SCALE* k_scale, const T_SCALE* v_scale,
    int num_heads, int kv_num_heads, int head_size, int d, int max_seqlen,
    const int* past_seq_lens, const T* cos_cache, const T* sin_cache,
    int rotary_dim, const int64_t* position_ids, bool interleaved, bool is_cache_bnsh, bool per_channel) {
  UnpackRoPEQuantizeAppend<T, T_QUANT, T_SCALE, bit_width><<<grid, block, 0, stream>>>(
      packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
      num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
      cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
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
  // Determine vectorization factor (float4 is 16 bytes)
  constexpr int vector_bytes = sizeof(float4);
  constexpr int element_bytes = sizeof(T);
  constexpr int elements_per_vector = vector_bytes / element_bytes;

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by ", elements_per_vector, " for vectorized GQA kernel.");
  }

  if (sequence_length > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Sequence length ", sequence_length, " exceeds CUDA grid limit (65535).");
  }

  const int d = (num_heads + 2 * kv_num_heads) * head_size;
  const int d_vectors = d / elements_per_vector;

  const int threads_per_block = std::min(max_threads_per_block, d_vectors);
  const int blocks_x = (d_vectors + threads_per_block - 1) / threads_per_block;
  const dim3 grid(blocks_x, sequence_length, batch_size);
  const dim3 block(threads_per_block);

  float* k_scale = nullptr;
  float* v_scale = nullptr;
  bool per_channel = false;

  DispatchUnpackRoPEQuantizeAppend<T, T, float, 16>(
      grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
      k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
      cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);

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
