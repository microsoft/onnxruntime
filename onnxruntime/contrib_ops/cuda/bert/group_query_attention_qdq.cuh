// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Enable quantized KV cache support for INT8/INT4
#define KV_QUANT_SUPPORTED 1

// #include <cstdio> // Added for printf
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

// Constants for quantization bounds
constexpr int kInt4Min = -8;
constexpr int kInt4Max = 7;
constexpr int kInt8Min = -128;
constexpr int kInt8Max = 127;
constexpr int kInt4ZeroPacked = 0x88;  // (0 + 8) | ((0 + 8) << 4) for INT4 zero padding
constexpr int kThreadsPerBlock = 256;

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
                                 const T_SCALE* scale, const int* past_seq_lens,
                                 int batch_size, int num_heads,
                                 int cache_sequence_length,
                                 int head_size, int bit_width,
                                 KVQuantizationType quant_type,
                                 bool is_input_bsnh) {
  int64_t total_elements = static_cast<int64_t>(batch_size) * num_heads * cache_sequence_length * head_size;
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total_elements;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int h = static_cast<int>(i % head_size);
    int s = static_cast<int>((i / head_size) % cache_sequence_length);
    int n = static_cast<int>((i / (head_size * cache_sequence_length)) % num_heads);
    int b = static_cast<int>((i / (num_heads * head_size * cache_sequence_length)));

    // Correctly identify padding in the past_kv cache.
    // In the decoding case, `seqlens` contains `past_len + new_len - 1`.
    // We need the actual past_len to mask the padding correctly.
    if (past_seq_lens != nullptr) {
      // For a given batch entry `b`, the actual length of the past sequence is `past_seq_lens[b]`.
      // If `s` (the current sequence index) is beyond this length, it's padding and should be zeroed.
      if (s >= past_seq_lens[b]) {
        dequantized_data[i] = static_cast<T>(0.0f);
        continue;
      }
    }

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = static_cast<float>(scale[0]);
    } else {  // PER_CHANNEL
      int64_t scale_idx = static_cast<int64_t>(n) * head_size + h;
      scale_val = static_cast<float>(scale[scale_idx]);
    }

    float quantized_float;
    // The input quantized_data is indexed by the actual cache sequence length (cache_sequence_length = cache_sequence_length)
    // not the output sequence length.
    int64_t input_idx = static_cast<int64_t>(b) * num_heads * cache_sequence_length * elements_per_head_packed +
                        static_cast<int64_t>(n) * cache_sequence_length * elements_per_head_packed +
                        static_cast<int64_t>(s) * elements_per_head_packed +
                        (bit_width == 4 ? h / 2 : h);

    if (is_input_bsnh) {
      input_idx = static_cast<int64_t>(b) * cache_sequence_length * num_heads * elements_per_head_packed +
                  static_cast<int64_t>(s) * num_heads * elements_per_head_packed +
                  static_cast<int64_t>(n) * elements_per_head_packed +
                  (bit_width == 4 ? h / 2 : h);
    }

    if (bit_width == 8) {
      quantized_float = static_cast<float>(
          reinterpret_cast<const int8_t*>(quantized_data)[input_idx]);
    } else {  // 4
      const uint8_t packed_val =
          reinterpret_cast<const uint8_t*>(quantized_data)[input_idx];
      quantized_float = (h % 2 == 0)
                            ? static_cast<float>((packed_val & 0x0F) - 8)
                            : static_cast<float>((packed_val >> 4) - 8);
    }

    dequantized_data[i] = static_cast<T>(quantized_float * scale_val);
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchDequantizeKV(cudaStream_t stream, T* dequantized_data,
                          const T_QUANT* quantized_data, const T_SCALE* scale,
                          const int* past_seq_lens, int batch_size, int num_heads,
                          int cache_sequence_length,
                          int head_size, int bit_width,
                          KVQuantizationType quant_type,
                          bool is_input_bsnh) {
  if (cache_sequence_length == 0) return Status::OK();

  // Output buffer uses cache_sequence_length stride
  int64_t total_elements = static_cast<int64_t>(batch_size) * num_heads * cache_sequence_length * head_size;
  const int blocks = static_cast<int>((total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
  DequantizeKernel<T, T_QUANT, T_SCALE><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dequantized_data, quantized_data, scale, past_seq_lens,
      batch_size, num_heads, cache_sequence_length,
      head_size, bit_width, quant_type, is_input_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

// Quantization Kernel for KV cache.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeKernel(T_QUANT* quantized_data,
                               const T* dequantized_data, const T_SCALE* scale,
                               const int* past_seq_lens,
                               const int* total_seq_lens,
                               int total_packed_elements,
                               int input_sequence_length,
                               int cache_sequence_length, int num_heads, int head_size,
                               int bit_width, KVQuantizationType quant_type,
                               bool is_input_bsnh,
                               bool is_output_bsnh) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total_packed_elements;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int h_packed = static_cast<int>(i % elements_per_head_packed);
    int s = static_cast<int>((i / elements_per_head_packed) % cache_sequence_length);
    int n = static_cast<int>((i / (elements_per_head_packed * cache_sequence_length)) % num_heads);
    int b = static_cast<int>(i / (num_heads * elements_per_head_packed * cache_sequence_length));

    // If past_seq_lens is provided, skip the past data to preserve it.
    // This is useful when we are appending new data to an existing quantized cache (shared buffer).
    if (past_seq_lens != nullptr) {
      if (s < past_seq_lens[b]) {
        continue;
      }
    }

    // Zero out padding in the present_kv cache.
    // `total_seq_lens` provides the total valid sequence length for each batch item.
    // If the current sequence index `s` is in the padded region, write zero.
    int total_valid_len_b = total_seq_lens[b];
    if (s >= total_valid_len_b) {
      if (bit_width == 8) {
        int64_t out_idx = i;
        if (is_output_bsnh) {
          int64_t b_idx = b;
          int64_t n_idx = n;
          int64_t s_idx = s;
          int64_t h_idx = i % elements_per_head_packed;
          out_idx = b_idx * cache_sequence_length * num_heads * elements_per_head_packed +
                    s_idx * num_heads * elements_per_head_packed +
                    n_idx * elements_per_head_packed +
                    h_idx;
        }
        reinterpret_cast<int8_t*>(quantized_data)[out_idx] = 0;
      } else {  // INT4
        // With packed iteration, each thread handles one byte (2 values).
        // Since we are in the padding region, write a zero byte.
        // For BNSH/BSNH output, we need to calculate correct index.
        int64_t out_idx = i;
        if (is_output_bsnh) {
          int64_t b_idx = b;
          int64_t n_idx = n;
          int64_t s_idx = s;
          int64_t h_idx = i % elements_per_head_packed;
          out_idx = b_idx * cache_sequence_length * num_heads * elements_per_head_packed +
                    s_idx * num_heads * elements_per_head_packed +
                    n_idx * elements_per_head_packed +
                    h_idx;
        }
        // INT4 uses +8 bias, so zero values pack to 0x88
        reinterpret_cast<uint8_t*>(quantized_data)[out_idx] = kInt4ZeroPacked;
      }
      continue;
    }

    int64_t output_idx = i;
    if (is_output_bsnh) {
      int64_t b_idx = b;
      int64_t n_idx = n;
      int64_t s_idx = s;
      int64_t h_idx = i % elements_per_head_packed;
      output_idx = b_idx * cache_sequence_length * num_heads * elements_per_head_packed +
                   s_idx * num_heads * elements_per_head_packed +
                   n_idx * elements_per_head_packed +
                   h_idx;
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
      int64_t flattened_input_idx = is_input_bsnh ? ((int64_t)b * input_sequence_length * num_heads * head_size +
                                                     (int64_t)s * num_heads * head_size +
                                                     (int64_t)n * head_size +
                                                     h)
                                                  : ((int64_t)b * num_heads * input_sequence_length * head_size +
                                                     (int64_t)n * input_sequence_length * head_size +
                                                     (int64_t)s * head_size +
                                                     h);
      float val_float = static_cast<float>(dequantized_data[flattened_input_idx]) * inv_scale;

      int32_t val_int32 = static_cast<int32_t>(rintf(val_float));
      reinterpret_cast<int8_t*>(quantized_data)[output_idx] =
          static_cast<int8_t>(max(kInt8Min, min(kInt8Max, val_int32)));
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

      int64_t input_idx0 = is_input_bsnh ? ((int64_t)b * input_sequence_length * num_heads * head_size +
                                            (int64_t)s * num_heads * head_size +
                                            (int64_t)n * head_size +
                                            h0)
                                         : ((int64_t)b * num_heads * input_sequence_length * head_size +
                                            (int64_t)n * input_sequence_length * head_size +
                                            (int64_t)s * head_size +
                                            h0);
      float val0 = static_cast<float>(dequantized_data[input_idx0]) * inv_scale0;
      int8_t q0 = static_cast<int8_t>(max(static_cast<float>(kInt4Min), min(static_cast<float>(kInt4Max), rintf(val0))));

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

        int64_t input_idx1 = is_input_bsnh ? ((int64_t)b * input_sequence_length * num_heads * head_size +
                                              (int64_t)s * num_heads * head_size +
                                              (int64_t)n * head_size +
                                              h1)
                                           : ((int64_t)b * num_heads * input_sequence_length * head_size +
                                              (int64_t)n * input_sequence_length * head_size +
                                              (int64_t)s * head_size +
                                              h1);
        float val1 = static_cast<float>(dequantized_data[input_idx1]) * inv_scale1;
        q1 = static_cast<int8_t>(max(static_cast<float>(kInt4Min), min(static_cast<float>(kInt4Max), rintf(val1))));
      } else {
        // Padding for odd head_size
        q1 = 0;
      }

      // Pack two 4-bit values into one byte with +8 bias to convert to unsigned [0,15]
      // Low nibble: q0 (even element), High nibble: q1 (odd element)
      uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
      reinterpret_cast<uint8_t*>(quantized_data)[output_idx] = packed;
    }
  }
}

// Append kernel for dynamic offset quantization
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeAppendKernel(T_QUANT* cache_data,
                                     const T* new_data,
                                     const T_SCALE* scale,
                                     const int* total_seq_lens,
                                     int max_seq_len,
                                     int num_heads,
                                     int head_size,
                                     int bit_width,
                                     int new_seq_len,
                                     KVQuantizationType quant_type,
                                     int batch_size,
                                     bool is_input_bsnh,
                                     bool is_output_bsnh) {
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

  int past_len = 0;
  if (total_seq_lens != nullptr) {
    past_len = total_seq_lens[b] - new_seq_len;
  }

  // For safety, ensure past_len >= 0. In prompt phase, it's 0.
  if (past_len < 0) past_len = 0;

  int64_t cache_offset;
  if (is_output_bsnh) {
    cache_offset = (int64_t)b * max_seq_len * num_heads * elements_per_head_packed +
                   (int64_t)(past_len + s) * num_heads * elements_per_head_packed +
                   (int64_t)n * elements_per_head_packed +
                   h_packed;
  } else {
    cache_offset = (int64_t)b * num_heads * max_seq_len * elements_per_head_packed +
                   (int64_t)n * max_seq_len * elements_per_head_packed +
                   (int64_t)(past_len + s) * elements_per_head_packed +
                   h_packed;
  }

  if (bit_width == 8) {
    int h = h_packed;
    int64_t src_idx = is_input_bsnh ? ((int64_t)b * new_seq_len * num_heads * head_size +
                                       (int64_t)s * num_heads * head_size +
                                       (int64_t)n * head_size +
                                       h)
                                    : ((int64_t)b * num_heads * new_seq_len * head_size +
                                       (int64_t)n * new_seq_len * head_size +
                                       (int64_t)s * head_size +
                                       h);
    float val = static_cast<float>(new_data[src_idx]);

    float s_scale = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR)
      s_scale = static_cast<float>(scale[0]);
    else if (quant_type == KVQuantizationType::PER_CHANNEL)
      s_scale = static_cast<float>(scale[n * head_size + h]);

    float inv_s = (s_scale == 0.0f) ? 0.0f : 1.0f / s_scale;
    int8_t q_val = static_cast<int8_t>(max(static_cast<float>(kInt8Min), min(static_cast<float>(kInt8Max), rintf(val * inv_s))));
    reinterpret_cast<int8_t*>(cache_data)[cache_offset] = q_val;

  } else {  // Int4
    int h0 = h_packed * 2;
    int h1 = h0 + 1;

    int64_t src_idx0 = is_input_bsnh ? ((int64_t)b * new_seq_len * num_heads * head_size +
                                        (int64_t)s * num_heads * head_size +
                                        (int64_t)n * head_size +
                                        h0)
                                     : ((int64_t)b * num_heads * new_seq_len * head_size +
                                        (int64_t)n * new_seq_len * head_size +
                                        (int64_t)s * head_size +
                                        h0);
    float val0 = static_cast<float>(new_data[src_idx0]);

    float val1 = 0.0f;
    if (h1 < head_size) {
      int64_t src_idx1 = is_input_bsnh ? ((int64_t)b * new_seq_len * num_heads * head_size +
                                          (int64_t)s * num_heads * head_size +
                                          (int64_t)n * head_size +
                                          h1)
                                       : ((int64_t)b * num_heads * new_seq_len * head_size +
                                          (int64_t)n * new_seq_len * head_size +
                                          (int64_t)s * head_size +
                                          h1);
      val1 = static_cast<float>(new_data[src_idx1]);
    }

    float s0 = 1.0f;
    float s1 = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      s0 = static_cast<float>(scale[0]);
      s1 = static_cast<float>(scale[0]);
    } else {
      s0 = static_cast<float>(scale[n * head_size + h0]);
      if (h1 < head_size) s1 = static_cast<float>(scale[n * head_size + h1]);
    }

    int8_t q0 = static_cast<int8_t>(max(static_cast<float>(kInt4Min), min(static_cast<float>(kInt4Max), rintf(val0 * (s0 == 0 ? 0 : 1.0f / s0)))));
    int8_t q1 = static_cast<int8_t>(max(static_cast<float>(kInt4Min), min(static_cast<float>(kInt4Max), rintf(val1 * (s1 == 0 ? 0 : 1.0f / s1)))));

    uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
    reinterpret_cast<uint8_t*>(cache_data)[cache_offset] = packed;
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeKV(cudaStream_t stream, T_QUANT* quantized_data,
                        const T* dequantized_data, const T_SCALE* scale,
                        const int* past_seq_lens,
                        const int* total_seq_lens,
                        int batch_size, int num_heads,
                        int input_sequence_length, int cache_sequence_length, int head_size, int bit_width,
                        KVQuantizationType quant_type,
                        bool is_input_bsnh,
                        bool is_output_bsnh) {
  assert(total_seq_lens != nullptr);
  if (cache_sequence_length == 0) return Status::OK();

  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_packed_elements = batch_size * num_heads * cache_sequence_length * elements_per_head_packed;

  int blocks = (total_packed_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;

  QuantizeKernel<T, T_QUANT, T_SCALE><<<blocks, kThreadsPerBlock, 0, stream>>>(
      quantized_data, dequantized_data, scale, past_seq_lens, total_seq_lens, total_packed_elements,
      input_sequence_length, cache_sequence_length, num_heads, head_size, bit_width, quant_type, is_input_bsnh, is_output_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeAppendKV(cudaStream_t stream, T_QUANT* cache_data,
                              const T* new_data, const T_SCALE* scale,
                              const int* total_seq_lens, int batch_size, int num_heads,
                              int max_seq_len, int head_size, int bit_width,
                              int new_seq_len,
                              KVQuantizationType quant_type,
                              bool is_input_bsnh,
                              bool is_output_bsnh) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_threads = batch_size * num_heads * new_seq_len * elements_per_head_packed;
  int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

  QuantizeAppendKernel<T, T_QUANT, T_SCALE><<<blocks, kThreadsPerBlock, 0, stream>>>(
      cache_data, new_data, scale, total_seq_lens, max_seq_len, num_heads, head_size, bit_width, new_seq_len, quant_type, batch_size, is_input_bsnh, is_output_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations for launchers
template Status LaunchDequantizeKV<half, int8_t, float>(
    cudaStream_t, half*, const int8_t*, const float*, const int*, int, int, int, int, int, KVQuantizationType, bool);
template Status LaunchDequantizeKV<half, uint8_t, float>(
    cudaStream_t, half*, const uint8_t*, const float*, const int*, int, int, int, int, int, KVQuantizationType, bool);
template Status LaunchDequantizeKV<BFloat16, int8_t, float>(
    cudaStream_t, BFloat16*, const int8_t*, const float*, const int*, int, int, int, int, int, KVQuantizationType, bool);
template Status LaunchDequantizeKV<BFloat16, uint8_t, float>(
    cudaStream_t, BFloat16*, const uint8_t*, const float*, const int*, int, int, int, int, int, KVQuantizationType, bool);

template Status LaunchQuantizeKV<half, int8_t, float>(
    cudaStream_t, int8_t*, const half*, const float*, const int*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeKV<half, uint8_t, float>(
    cudaStream_t, uint8_t*, const half*, const float*, const int*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeKV<BFloat16, int8_t, float>(
    cudaStream_t, int8_t*, const BFloat16*, const float*, const int*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeKV<BFloat16, uint8_t, float>(
    cudaStream_t, uint8_t*, const BFloat16*, const float*, const int*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);

template Status LaunchQuantizeAppendKV<half, int8_t, float>(
    cudaStream_t, int8_t*, const half*, const float*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeAppendKV<half, uint8_t, float>(
    cudaStream_t, uint8_t*, const half*, const float*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeAppendKV<BFloat16, int8_t, float>(
    cudaStream_t, int8_t*, const BFloat16*, const float*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);
template Status LaunchQuantizeAppendKV<BFloat16, uint8_t, float>(
    cudaStream_t, uint8_t*, const BFloat16*, const float*, const int*, int, int, int, int, int, int, KVQuantizationType, bool, bool);

// Fused kernel: Unpack QKV + Apply RoPE to Q and K + Append K/V directly to cache
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
Status LaunchUnpackRoPEQuantizeAppend(
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, void* k_cache, void* v_cache,
    const float* k_scale, const float* v_scale,
    const int num_heads, const int kv_num_heads, const int head_size,
    const int sequence_length, const int batch_size, const int max_seqlen,
    const int* past_seq_lens, const T* cos_cache, const T* sin_cache,
    const int rotary_dim, const int64_t* position_ids, const bool interleaved,
    const bool is_cache_bnsh, const KVQuantizationType k_quant_type,
    const int bit_width, cudaStream_t stream, const int max_threads_per_block) {
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

  bool per_channel = (k_quant_type == KVQuantizationType::PER_CHANNEL);

  if (bit_width == 16 || bit_width == 32) {
    DispatchUnpackRoPEQuantizeAppend<T, T, float, 16>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, static_cast<T*>(k_cache), static_cast<T*>(v_cache),
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (bit_width == 8) {
    DispatchUnpackRoPEQuantizeAppend<T, int8_t, float, 8>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, static_cast<int8_t*>(k_cache), static_cast<int8_t*>(v_cache),
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (bit_width == 4) {
    DispatchUnpackRoPEQuantizeAppend<T, uint8_t, float, 4>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, static_cast<uint8_t*>(k_cache), static_cast<uint8_t*>(v_cache),
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit template instantiations
template Status LaunchUnpackRoPEQuantizeAppend<half>(
    const half*, const half*, const half*, const half*, half*, void*, void*, const float*, const float*,
    int, int, int, int, int, int, const int*, const half*, const half*, int, const int64_t*, bool, bool,
    KVQuantizationType, int, cudaStream_t, int);

template Status LaunchUnpackRoPEQuantizeAppend<BFloat16>(
    const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, BFloat16*, void*, void*, const float*, const float*,
    int, int, int, int, int, int, const int*, const BFloat16*, const BFloat16*, int, const int64_t*, bool, bool,
    KVQuantizationType, int, cudaStream_t, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
