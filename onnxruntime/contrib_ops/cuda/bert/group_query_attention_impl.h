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

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data);

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
//   position_ids_buffer = GetScratchBuffer<void>(req.position_ids_bytes, ...);
// ============================================================================
struct GQABufferRequirements {
  size_t unpacked_qkv_bytes = 0;
  size_t rotary_buffer_bytes = 0;
  size_t position_ids_bytes = 0;

  template <typename T>
  static GQABufferRequirements Compute(
      const GroupQueryAttentionParameters& params,
      bool use_flash_attention,
      bool use_flash_attention_fast_decode,
      bool use_memory_efficient_attention) {
    GQABufferRequirements req;

    const size_t elem_size = sizeof(T);
    const size_t batch_size = static_cast<size_t>(params.batch_size);
    const size_t seq_len = static_cast<size_t>(params.sequence_length);
    const size_t num_heads = static_cast<size_t>(params.num_heads);
    const size_t kv_num_heads = static_cast<size_t>(params.kv_num_heads);
    const size_t head_size = static_cast<size_t>(params.head_size);

    // Fast decode path: Flash Attention handles everything internally
    if (use_flash_attention_fast_decode) {
      return req;  // All zeros - no scratch buffers needed
    }

    // Q, K, V element counts
    const size_t q_elements = batch_size * seq_len * num_heads * head_size;
    const size_t k_elements = batch_size * seq_len * kv_num_heads * head_size;
    const size_t v_elements = k_elements;

    if (use_flash_attention) {
      // Flash Attention path:
      // - unpacked_qkv_buffer is used for:
      //   1. Unpacking packed QKV input
      //   2. Storing rotated Q (and K for non-fused path)
      // - rotary_buffer is NOT used (rotations go to unpacked_qkv_buffer)
      // - position_ids_buffer is NOT used (flash attention uses implicit position IDs)

      if (params.is_packed_qkv) {
        // Need full Q+K+V for unpacking
        req.unpacked_qkv_bytes = elem_size * (q_elements + k_elements + v_elements);
      } else if (params.do_rotary) {
        // Unpacked input with RoPE: need Q+K for rotation output
        req.unpacked_qkv_bytes = elem_size * (q_elements + k_elements);
      }
      // Note: unpacked + no-RoPE case does NOT need unpacked_qkv_buffer

    } else if (use_memory_efficient_attention) {
      // Memory Efficient Attention path:
      // - unpacked_qkv_buffer: for unpacking packed QKV
      // - rotary_buffer: for Q and K rotation output (separate from unpack buffer)
      // - position_ids_buffer: for explicit position IDs if needed

      if (params.is_packed_qkv) {
        req.unpacked_qkv_bytes = elem_size * (q_elements + k_elements + v_elements);
      }

      if (params.do_rotary) {
        // Q rotation + K rotation
        // Note: K uses kv_num_heads which may be less than num_heads
        req.rotary_buffer_bytes = elem_size * (q_elements + k_elements);
        // Position IDs space (always allocated for MEA + RoPE path)
        req.position_ids_bytes = sizeof(int64_t) * batch_size * seq_len;
      }
    }

    return req;
  }
};

// ============================================================================
// Debug helper for tracking buffer usage
// ============================================================================
// Call these after buffer access to record the maximum offset used.
// In release builds, these are no-ops.
//
// Example:
//   T* unpacked_q = data.unpacked_qkv_buffer;
//   // ... kernel writes to unpacked_q[0..Q_size-1] ...
//   UpdateUnpackedQkvMaxUsed(data, Q_size * sizeof(T));
// ============================================================================
#ifndef NDEBUG
template <typename T>
inline void UpdateUnpackedQkvMaxUsed(GroupQueryAttentionData<T>& data, size_t bytes_used) {
  if (bytes_used > data.unpacked_qkv_max_used) {
    data.unpacked_qkv_max_used = bytes_used;
  }
}

template <typename T>
inline void UpdateRotaryMaxUsed(GroupQueryAttentionData<T>& data, size_t bytes_used) {
  if (bytes_used > data.rotary_max_used) {
    data.rotary_max_used = bytes_used;
  }
}

template <typename T>
inline void UpdatePositionIdsMaxUsed(GroupQueryAttentionData<T>& data, size_t bytes_used) {
  if (bytes_used > data.position_ids_max_used) {
    data.position_ids_max_used = bytes_used;
  }
}
#else
template <typename T>
inline void UpdateUnpackedQkvMaxUsed(GroupQueryAttentionData<T>&, size_t) {}
template <typename T>
inline void UpdateRotaryMaxUsed(GroupQueryAttentionData<T>&, size_t) {}
template <typename T>
inline void UpdatePositionIdsMaxUsed(GroupQueryAttentionData<T>&, size_t) {}
#endif

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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
