// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// #include <cstdio> // Added for printf
#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ============================================================================
// KV Cache Quantization/Dequantization Kernels
// ============================================================================
//
// This file implements symmetric quantization for KV cache in GroupQueryAttention.
// Supports INT4 and INT8 with PER_TENSOR and PER_CHANNEL quantization modes.
//
// QUANTIZATION SCHEME:
// -------------------
// INT4: Symmetric signed quantization
//   - Range: [-8, 7] (signed 4-bit)
//   - Formula: q = clamp(round(x / scale), -8, 7)
//   - Rounding: Round-to-nearest (rintf)
//   - Saturation: Clamp to [-8, 7]
//
// INT8: Symmetric signed quantization
//   - Range: [-128, 127] (signed 8-bit)
//   - Formula: q = clamp(round(x / scale), -128, 127)
//   - Rounding: Round-to-nearest (rintf)
//   - Saturation: Clamp to [-128, 127]
//
// BIT PACKING (INT4 only):
// -----------------------
// Storage format: uint8_t, 2 values per byte
//   packed_byte = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4)
//
// Where:
//   - q0 (even element) → low nibble (bits 0-3)
//   - q1 (odd element) → high nibble (bits 4-7)
//   - +8 bias converts signed [-8, 7] to unsigned [0, 15]
//
// For odd head_size, last element q0 is paired with q1 = 0.
//
// SCALE TENSOR FORMAT:
// -------------------
// Scales are always FP16/BF16 (type T), never quantized.
//
// PER_TENSOR: scale[0] - single scale for entire cache
// PER_CHANNEL: scale[head_idx * head_size + elem_idx] - one scale per channel
//
// MEMORY LAYOUT:
// -------------
// Cache: BNSH (batch, num_heads, sequence_length, head_size)
// INT4: (head_size + 1) / 2 bytes per head
// INT8: head_size bytes per head
// ============================================================================

// Dequantization Kernel for KV cache.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void DequantizeKernel(T* dequantized_data,
                                 const T_QUANT* quantized_data,
                                 const T_SCALE* scale, const int* seqlens,
                                 int batch_size, int num_heads,
                                 int cache_sequence_length, int sequence_length,
                                 int head_size, bool is_past, int bit_width,
                                 KVQuantizationType quant_type) {
  int S = cache_sequence_length;
  int total_elements = batch_size * num_heads * S * head_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    int h = i % head_size;
    int s = (i / head_size) % S;
    int n = (i / (head_size * S)) % num_heads;
    int b = i / (num_heads * head_size * S);

    // Correctly identify padding in the past_kv cache.
    // In the decoding case, `seqlens` contains `past_len + new_len - 1`.
    // We need the actual past_len to mask the padding correctly.
    if (is_past && seqlens != nullptr) {
      // For a given batch entry `b`, the actual length of the past sequence is `seqlens[b] + 1 - sequence_length`.
      // If `s` (the current sequence index) is beyond this length, it's padding and should be zeroed.
      int past_len_b = seqlens[b] + 1 - sequence_length;
      if (s >= past_len_b) {
        dequantized_data[i] = static_cast<T>(0.0f);
        continue;
      }
    }

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = static_cast<float>(scale[0]);
    } else {  // PER_CHANNEL
      int scale_idx = n * head_size + h;
      scale_val = static_cast<float>(scale[scale_idx]);
    }

    float quantized_float;
    if (bit_width == 8) {
      quantized_float = static_cast<float>(
          reinterpret_cast<const int8_t*>(quantized_data)[i]);
    } else {  // 4
      const uint8_t packed_val =
          reinterpret_cast<const uint8_t*>(quantized_data)[i / 2];
      quantized_float = (i % 2 == 0)
                            ? static_cast<float>((packed_val & 0x0F) - 8)
                            : static_cast<float>((packed_val >> 4) - 8);
    }

    dequantized_data[i] = static_cast<T>(quantized_float * scale_val);
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchDequantizeKV(cudaStream_t stream, T* dequantized_data,
                          const T_QUANT* quantized_data, const T_SCALE* scale,
                          const int* seqlens, int batch_size, int num_heads,
                          int cache_sequence_length, int sequence_length,
                          int head_size, bool is_past, int bit_width,
                          KVQuantizationType quant_type) {
  int S = cache_sequence_length;
  if (S == 0) return Status::OK();

  int total_elements = batch_size * num_heads * S * head_size;
  const int threads_per_block = 256;
  const int blocks =
      (total_elements + threads_per_block - 1) / threads_per_block;

  DequantizeKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      dequantized_data, quantized_data, scale, seqlens, batch_size, num_heads,
      cache_sequence_length, sequence_length, head_size, is_past, bit_width,
      quant_type);

  return CUDA_CALL(cudaGetLastError());
}

// Quantization Kernel for KV cache.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeKernel(T_QUANT* quantized_data,
                               const T* dequantized_data, const T_SCALE* scale,
                               const int* seqlens, int total_packed_elements,
                               int cache_sequence_length, int num_heads, int head_size,
                               int bit_width, KVQuantizationType quant_type) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_packed_elements;
       i += blockDim.x * gridDim.x) {
    int h_packed = i % elements_per_head_packed;
    int s = (i / elements_per_head_packed) % cache_sequence_length;
    int n = (i / (elements_per_head_packed * cache_sequence_length)) % num_heads;
    int b = i / (num_heads * elements_per_head_packed * cache_sequence_length);

    // Zero out padding in the present_kv cache.
    // `seqlens` (seqlens_k) provides the total valid sequence length for each batch item.
    // If the current sequence index `s` is in the padded region, write zero.
    int total_valid_len_b = seqlens[b] + 1;
    if (s >= total_valid_len_b) {
      if (bit_width == 8) {
        reinterpret_cast<int8_t*>(quantized_data)[i] = 0;
      } else {  // 4
        // With packed iteration, each thread handles one byte (2 values).
        // Since we are in the padding region, write a zero byte.
        reinterpret_cast<uint8_t*>(quantized_data)[i] = (0 + 8) | ((0 + 8) << 4);
      }
      continue;
    }

    if (bit_width == 8) {
      int h = h_packed;
      float scale_val = 1.0f;
      if (quant_type == KVQuantizationType::PER_TENSOR) {
        scale_val = static_cast<float>(scale[0]);
      } else {  // PER_CHANNEL
        int scale_idx = n * head_size + h;
        scale_val = static_cast<float>(scale[scale_idx]);
      }

      float inv_scale = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
      int64_t flattened_input_idx = (int64_t)b * num_heads * cache_sequence_length * head_size +
                                    (int64_t)n * cache_sequence_length * head_size +
                                    (int64_t)s * head_size +
                                    h;
      float val_float = static_cast<float>(dequantized_data[flattened_input_idx]) * inv_scale;

      int32_t val_int32 = static_cast<int32_t>(rintf(val_float));
      reinterpret_cast<int8_t*>(quantized_data)[i] =
          static_cast<int8_t>(max(-128, min(127, val_int32)));
    } else {  // 4
      int h0 = h_packed * 2;
      int h1 = h0 + 1;

      // Compute first nibble
      float scale0 = 1.0f;
      if (quant_type == KVQuantizationType::PER_TENSOR) {
        scale0 = static_cast<float>(scale[0]);
      } else {
        scale0 = static_cast<float>(scale[n * head_size + h0]);
      }
      float inv_scale0 = (scale0 == 0.0f) ? 0.0f : 1.0f / scale0;

      int64_t input_idx0 = (int64_t)b * num_heads * cache_sequence_length * head_size +
                           (int64_t)n * cache_sequence_length * head_size +
                           (int64_t)s * head_size +
                           h0;
      float val0 = static_cast<float>(dequantized_data[input_idx0]) * inv_scale0;
      int8_t q0 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val0))));

      // Compute second nibble if within head_size
      int8_t q1 = 0;  // Default to 0 (value 0) if padded
      if (h1 < head_size) {
        float scale1 = 1.0f;
        if (quant_type == KVQuantizationType::PER_TENSOR) {
          scale1 = static_cast<float>(scale[0]);
        } else {
          scale1 = static_cast<float>(scale[n * head_size + h1]);
        }
        float inv_scale1 = (scale1 == 0.0f) ? 0.0f : 1.0f / scale1;

        int64_t input_idx1 = (int64_t)b * num_heads * cache_sequence_length * head_size +
                             (int64_t)n * cache_sequence_length * head_size +
                             (int64_t)s * head_size +
                             h1;
        float val1 = static_cast<float>(dequantized_data[input_idx1]) * inv_scale1;
        q1 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val1))));
      } else {
        // Padding for odd head_size
        q1 = 0;
      }

      // Pack two 4-bit values into one byte with +8 bias to convert to unsigned [0,15]
      // Low nibble: q0 (even element), High nibble: q1 (odd element)
      uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
      reinterpret_cast<uint8_t*>(quantized_data)[i] = packed;
    }
  }
}

// Append kernel for dynamic offset quantization
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeAppendKernel(T_QUANT* cache_data,
                                     const T* new_data,
                                     const T_SCALE* scale,
                                     const int* total_seqlens,
                                     int max_seq_len,
                                     int num_heads,
                                     int head_size,
                                     int bit_width,
                                     int new_seq_len,
                                     KVQuantizationType quant_type,
                                     int batch_size) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * num_heads * new_seq_len * elements_per_head_packed;

  if (idx >= total_elements) return;

  int h_packed = idx % elements_per_head_packed;
  int tmp = idx / elements_per_head_packed;
  int s = tmp % new_seq_len;
  tmp = tmp / new_seq_len;
  int n = tmp % num_heads;
  int b = tmp / num_heads;

  if (b >= batch_size) return;

  int past_len = 0;
  if (total_seqlens != nullptr) {
    past_len = total_seqlens[b] - new_seq_len;
  }
  // For safety, ensure past_len >= 0. In prompt phase, it's 0.
  if (past_len < 0) past_len = 0;

  int64_t cache_offset = (int64_t)b * num_heads * max_seq_len * elements_per_head_packed +
                         (int64_t)n * max_seq_len * elements_per_head_packed +
                         (int64_t)(past_len + s) * elements_per_head_packed +
                         h_packed;

  if (bit_width == 8) {
    int h = h_packed;
    int64_t src_idx = (int64_t)b * num_heads * new_seq_len * head_size +
                      (int64_t)n * new_seq_len * head_size +
                      (int64_t)s * head_size +
                      h;
    float val = static_cast<float>(new_data[src_idx]);

    float s_scale = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR)
      s_scale = (float)scale[0];
    else if (quant_type == KVQuantizationType::PER_CHANNEL)
      s_scale = (float)scale[n * head_size + h];

    float inv_s = (s_scale == 0.0f) ? 0.0f : 1.0f / s_scale;
    int8_t q_val = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(val * inv_s))));
    reinterpret_cast<int8_t*>(cache_data)[cache_offset] = q_val;

  } else {  // Int4
    int h0 = h_packed * 2;
    int h1 = h0 + 1;

    int64_t src_idx0 = (int64_t)b * num_heads * new_seq_len * head_size +
                       (int64_t)n * new_seq_len * head_size +
                       (int64_t)s * head_size +
                       h0;
    float val0 = static_cast<float>(new_data[src_idx0]);

    float val1 = 0.0f;
    if (h1 < head_size) {
      int64_t src_idx1 = (int64_t)b * num_heads * new_seq_len * head_size +
                         (int64_t)n * new_seq_len * head_size +
                         (int64_t)s * head_size +
                         h1;
      val1 = static_cast<float>(new_data[src_idx1]);
    }

    float s0 = 1.0f;
    float s1 = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      s0 = (float)scale[0];
      s1 = (float)scale[0];
    } else {
      s0 = (float)scale[n * head_size + h0];
      if (h1 < head_size) s1 = (float)scale[n * head_size + h1];
    }

    int8_t q0 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val0 * (s0 == 0 ? 0 : 1.0f / s0)))));
    int8_t q1 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val1 * (s1 == 0 ? 0 : 1.0f / s1)))));

    uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
    reinterpret_cast<uint8_t*>(cache_data)[cache_offset] = packed;
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeKV(cudaStream_t stream, T_QUANT* quantized_data,
                        const T* dequantized_data, const T_SCALE* scale,
                        const int* seqlens, int batch_size, int num_heads,
                        int cache_sequence_length, int head_size, int bit_width,
                        KVQuantizationType quant_type) {
  if (cache_sequence_length == 0) return Status::OK();

  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_packed_elements = batch_size * num_heads * cache_sequence_length * elements_per_head_packed;

  const int threads_per_block = 256;
  int blocks = (total_packed_elements + threads_per_block - 1) / threads_per_block;

  QuantizeKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      quantized_data, dequantized_data, scale, seqlens, total_packed_elements,
      cache_sequence_length, num_heads, head_size, bit_width, quant_type);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeAppendKV(cudaStream_t stream, T_QUANT* cache_data,
                              const T* new_data, const T_SCALE* scale,
                              const int* past_seqlens, int batch_size, int num_heads,
                              int max_seq_len, int head_size, int bit_width,
                              int new_seq_len,
                              KVQuantizationType quant_type) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_threads = batch_size * num_heads * new_seq_len * elements_per_head_packed;
  const int threads_per_block = 256;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  QuantizeAppendKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      cache_data, new_data, scale, past_seqlens, max_seq_len, num_heads, head_size, bit_width, new_seq_len, quant_type, batch_size);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations for launchers
template Status LaunchDequantizeKV<half, int8_t, half>(cudaStream_t, half*,
                                                       const int8_t*, const half*,
                                                       const int*, int, int, int, int,
                                                       int, bool, int,
                                                       KVQuantizationType);
template Status LaunchDequantizeKV<half, uint8_t, half>(cudaStream_t, half*,
                                                        const uint8_t*, const half*,
                                                        const int*, int, int, int,
                                                        int, int, bool, int,
                                                        KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, BFloat16*, const int8_t*, const BFloat16*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, BFloat16*, const uint8_t*, const BFloat16*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);

template Status LaunchQuantizeKV<half, int8_t, half>(
    cudaStream_t, int8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeKV<half, uint8_t, half>(
    cudaStream_t, uint8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, int8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, uint8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);

template Status LaunchQuantizeAppendKV<half, int8_t, half>(
    cudaStream_t, int8_t*, const half*, const half*, const int*, int, int, int, int,
    int, int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<half, uint8_t, half>(
    cudaStream_t, uint8_t*, const half*, const half*, const int*, int, int, int, int,
    int, int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, int8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, uint8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, int, KVQuantizationType);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
