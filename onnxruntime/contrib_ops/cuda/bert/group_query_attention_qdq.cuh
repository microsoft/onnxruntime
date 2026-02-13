// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Enable quantized KV cache support for INT8/INT4
#define KV_QUANT_SUPPORTED 1

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

template <typename T>
struct TypeConverter {
  __device__ static float to_float(T val) { return static_cast<float>(val); }
};

template <>
struct TypeConverter<half> {
  __device__ static float to_float(half val) { return __half2float(val); }
};

template <>
struct TypeConverter<__nv_bfloat16> {
  __device__ static float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
};

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

// Dequantization Kernel: Converts Quantized (Int8/Int4) KV cache back to Floating Point (T).
// Iterates over every individual element with one thread per element.
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
  // For BIT_WIDTH=4, each T_QUANT (uint8) holds 2 elements.
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
#ifdef USE_INT4_KV_CACHE
    } else if (bit_width == 4) {
      const uint8_t packed_val =
          reinterpret_cast<const uint8_t*>(quantized_data)[input_idx];
      quantized_float = (h % 2 == 0)
                            ? static_cast<float>((packed_val & 0x0F) - 8)
                            : static_cast<float>((packed_val >> 4) - 8);
#endif
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

// Quantization Kernel: Converts Floating Point (T) cache to Quantized (Int8/Int4) values.
// Note: This kernel is used to quantize a full input tensor, e.g. during graph initialization
// or fallback paths. The main prompt path uses the fused UnpackRoPEAppend kernel.
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
  // elements_per_head_packed is the number of BYTES occupied by head_size elements.
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
#ifdef USE_INT4_KV_CACHE
      } else if (bit_width == 4) {  // INT4
        // With packed iteration, each thread handles one byte (2 values).
        // Since we are in the padding region, write a zero byte.
        // For BNSH/BSNH output, we need to calculate correct index.
        // Memory Safety:
        // We iterate up to `total_packed_elements` which matches the allocated buffer size
        // (batch_size * num_heads * cache_sequence_length * elements_per_head_packed).
        // Since `h_idx` comes from `i % elements_per_head_packed`, `out_idx` is guaranteed
        // to be within the buffer bounds. Writing kInt4ZeroPacked is safe.
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
#endif
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
#ifdef USE_INT4_KV_CACHE
    } else if (bit_width == 4) {
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
#endif
    }
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
