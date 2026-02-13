// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data);

template <typename T, bool output_bnsh>
Status LaunchUnpackQKV(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
                       const int kv_num_heads, const int head_size, const int sequence_length, const int batch_size,
                       cudaStream_t stream, const int max_threads_per_block);

// ============================================================================
// GQABufferRequirements: Centralized buffer size calculation
// ============================================================================
// This struct provides a single source of truth for scratch buffer allocation.
// It ensures allocation logic in group_query_attention.cc stays in sync with
// kernel usage in group_query_attention_impl.cu.
//
// Usage:
//   auto req = GQABufferRequirements::Compute<T>(params, use_flash, fast_decode, use_mea, disable_fused);
//   unpacked_qkv_buffer = GetScratchBuffer<void>(req.unpacked_qkv_bytes, ...);
//   rotary_buffer = GetScratchBuffer<void>(req.rotary_buffer_bytes, ...);
// ============================================================================
struct GQABufferRequirements {
  size_t qkv_buffer_bytes = 0;

  template <typename T>
  static GQABufferRequirements Compute(
      const GroupQueryAttentionParameters& params,
      bool use_xqa,
      bool use_flash_attention,
      bool use_flash_attention_fast_decode,
      bool use_memory_efficient_attention) {
    GQABufferRequirements req;
    if (use_flash_attention_fast_decode) {
      return req;  // All zeros - no scratch buffers needed
    }

    const size_t elem_size = sizeof(T);
    const size_t batch_size = static_cast<size_t>(params.batch_size);
    const size_t seq_len = static_cast<size_t>(params.sequence_length);
    const size_t num_heads = static_cast<size_t>(params.num_heads);
    const size_t kv_num_heads = static_cast<size_t>(params.kv_num_heads);
    const size_t head_size = static_cast<size_t>(params.head_size);

    // Base requirements for all paths
    const size_t q_elements = batch_size * seq_len * num_heads * head_size;
    const size_t k_elements = batch_size * seq_len * kv_num_heads * head_size;
    const size_t v_elements = k_elements;

    if (use_xqa) {
      if (params.do_rotary || params.is_packed_qkv) {
        // XQA need scratch for rotated/unpacked Q.
        // RoPE K is written directly to cache by the fused kernel.
        req.qkv_buffer_bytes = elem_size * q_elements;
      }
      return req;
    }

    if (use_flash_attention) {
      // Flash Attention path:
      // qkv_buffer is used for:
      //   1. Unpacking packed Q (and K/V if needed)
      //   2. Storing rotated Q
      //
      // Logic:
      // - we generally only need Q buffer (for rotary Q) if we can write K/V directly to cache/output.

      bool is_quantized = params.k_quant_type != KVQuantizationType::NONE ||
                          params.v_quant_type != KVQuantizationType::NONE;

      if (is_quantized) {
        if (!params.is_first_prompt) {
          // Decoding fallback: need full cache scratch for dequantization
          // We need space for Q (rotated) + K (dequantized full) + V (dequantized full)
          // Q is sequence_length (1), K/V are seqlen_present_kv_cache (Capacity)
          const size_t k_elements_full = batch_size * static_cast<size_t>(params.seqlen_present_kv_cache) * kv_num_heads * head_size;
          // Align to 256 bytes for good measure
          size_t total_bytes = elem_size * (q_elements + 2 * k_elements_full) + 256;
          req.qkv_buffer_bytes = total_bytes;
        } else {
          req.qkv_buffer_bytes = elem_size * (q_elements + k_elements + v_elements);
        }
      } else if (params.do_rotary || params.is_packed_qkv) {
        req.qkv_buffer_bytes = elem_size * q_elements;
      }

    } else if (use_memory_efficient_attention) {
      // Memory Efficient Attention path:
      // - qkv_buffer: for unpacking packed QKV or Q rotation
      // MEA path usually needs Q, and also K, V if they need unpacking.
      // Current MEA implementation can handle separate K/V, but if packed, we unpack all.

      if (params.is_packed_qkv) {
        req.qkv_buffer_bytes = elem_size * (q_elements + k_elements + v_elements);
      } else if (params.do_rotary) {
        // Q rotation + K rotation
        req.qkv_buffer_bytes = elem_size * (q_elements + k_elements);
      }
    }

    return req;
  }
};

Status LaunchGetSequenceLengths(
    const int* total_seq_lens_minus_one,
    int* past_seq_lens,
    int* total_seq_lens,
    int* padded_seq_lens,
    const int batch_size,
    const int sequence_length,
    const bool is_first_prompt,
    cudaStream_t stream,
    const int max_threads_per_block);

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
    const int bit_width, cudaStream_t stream, const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
