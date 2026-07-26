// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Enable quantized KV cache support for INT8/INT4/FP8
#define KV_QUANT_SUPPORTED 1

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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
constexpr float kFp8E4M3Max = 448.0f;  // Max value for E4M3 format
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
// Supports INT4, INT8, and FP8 (E4M3) with PER_TENSOR and PER_CHANNEL quantization modes.
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
// INT8/FP8: head_size bytes per head
//
// FP8 E4M3: Native CUDA FP8 format
//   - Range: [-448, 448]
//   - Storage: __nv_fp8_e4m3 (1 byte)
//   - Conversion: Native CUDA cast via __nv_cvt_float_to_fp8/fp8_to_float
// ============================================================================

// Number of cache rows that have to be dequantized for a sequence of `valid_len` tokens.
// Rounded up to kDequantRowAlign so that a consumer which loads whole KV tiles (flash attention
// pads the final tile) still sees the exact same values the full-cache dequantization produced.
constexpr int kDequantRowAlign = 128;

__device__ __forceinline__ int ValidRowLimit(int valid_len, int cache_sequence_length) {
  const int rounded = (valid_len + kDequantRowAlign - 1) / kDequantRowAlign * kDequantRowAlign;
  return rounded < cache_sequence_length ? rounded : cache_sequence_length;
}

// Dequantization Kernel: Converts Quantized (Int8/Int4/FP8) KV cache back to Floating Point (T).
// Iterates over every individual element with one thread per element.
//
// `valid_seq_lens` (optional) is the per-batch total KV length. Rows at or beyond it are padding
// that the consumer (flash attention, which is given the same lengths) never reads, so they are
// left untouched instead of being dequantized. See DequantizeKVVectorizedKernel below for the fast
// path used by 8-bit caches.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void DequantizeKernel(T* dequantized_data,
                                 const T_QUANT* quantized_data,
                                 const T_SCALE* scale, const int* past_seq_lens,
                                 const int* valid_seq_lens,
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

    // Skip the padding tail of the cache: it is not part of the attended window.
    if (valid_seq_lens != nullptr && s >= ValidRowLimit(valid_seq_lens[b], cache_sequence_length)) {
      continue;
    }

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

    // FP8 check must come first since it also has bit_width=8
#ifdef USE_FP8_KV_CACHE
    if constexpr (std::is_same<T_QUANT, __nv_fp8_e4m3>::value) {
      __nv_fp8_e4m3 fp8_val = reinterpret_cast<const __nv_fp8_e4m3*>(quantized_data)[input_idx];
      quantized_float = static_cast<float>(fp8_val);
    } else
#endif
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

// ============================================================================
// Vectorized dequantization fast path for 8-bit caches (INT8 / FP8 E4M3).
//
// During decoding the whole KV cache is dequantized on every step, which makes this the dominant
// cost of the quantized-KV decode path. The generic kernel above spends it on scalar 1-byte loads,
// scalar 2-byte stores and four 64-bit div/mod per element. This variant instead
//   * gives each thread kVecSize contiguous head-size elements so loads and stores are 8/16 bytes,
//   * maps (batch, head) to blockIdx.y so the inner loop only does 32-bit address arithmetic,
//   * hoists the per-channel scales out of the sequence loop (they only depend on head/channel), and
//   * walks only the rows that are actually part of the sequence instead of the padded capacity.
//
// The grid is derived from the cache capacity only (never from a host-side sequence length) so the
// launch stays valid when the decode step is captured into a CUDA graph and replayed at other
// sequence lengths; the per-batch row limit is read from device memory inside the kernel.
// ============================================================================
template <int kBytes>
struct DequantVecType;
template <>
struct DequantVecType<4> {
  using type = uint32_t;
};
template <>
struct DequantVecType<8> {
  using type = uint2;
};
template <>
struct DequantVecType<16> {
  using type = uint4;
};
// 32 bytes is not a native load width; the compiler lowers this to two independent 16-byte loads,
// which keeps more memory requests in flight per thread.
struct alignas(16) DequantVec32 {
  uint4 lo;
  uint4 hi;
};
template <>
struct DequantVecType<32> {
  using type = DequantVec32;
};

template <typename T, typename T_QUANT, typename T_SCALE, int kVecSize>
__global__ void DequantizeKVVectorizedKernel(T* __restrict__ dequantized_data,
                                             const T_QUANT* __restrict__ quantized_data,
                                             const T_SCALE* __restrict__ scale,
                                             const int* __restrict__ valid_seq_lens,
                                             int num_heads,
                                             int cache_sequence_length,
                                             int head_size,
                                             KVQuantizationType quant_type,
                                             bool is_input_bsnh) {
  static_assert(sizeof(T_QUANT) == 1, "Vectorized dequantization only supports 8-bit caches.");
  static_assert(sizeof(T) == 2, "Vectorized dequantization only supports 16-bit outputs.");

  using LoadVec = typename DequantVecType<kVecSize>::type;
  constexpr int kStoreBytes = (kVecSize * 2 >= 16) ? 16 : kVecSize * 2;
  using StoreVec = typename DequantVecType<kStoreBytes>::type;
  constexpr int kStoresPerVec = (kVecSize * 2) / kStoreBytes;
  constexpr int kElemsPerStore = kVecSize / kStoresPerVec;

  const int vecs_per_row = head_size / kVecSize;
  const int rows_per_block = blockDim.x / vecs_per_row;
  const int vec_in_row = threadIdx.x % vecs_per_row;
  const int row_in_block = threadIdx.x / vecs_per_row;
  if (row_in_block >= rows_per_block) {
    return;  // blockDim.x is not an exact multiple of vecs_per_row
  }

  const int bn = blockIdx.y;  // b * num_heads + n
  const int n = bn % num_heads;
  const int b = bn / num_heads;

  const int limit = (valid_seq_lens == nullptr)
                        ? cache_sequence_length
                        : ValidRowLimit(valid_seq_lens[b], cache_sequence_length);

  const int h0 = vec_in_row * kVecSize;

  float scales[kVecSize];
  if (quant_type == KVQuantizationType::PER_TENSOR) {
    const float s0 = static_cast<float>(scale[0]);
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      scales[i] = s0;
    }
  } else {
    const T_SCALE* channel_scale = scale + static_cast<int64_t>(n) * head_size + h0;
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      scales[i] = static_cast<float>(channel_scale[i]);
    }
  }

  // The output is always BNSH; only the input row stride differs between BNSH and BSNH.
  const int64_t out_base = static_cast<int64_t>(bn) * cache_sequence_length * head_size + h0;
  const int64_t in_base = is_input_bsnh
                              ? (static_cast<int64_t>(b) * cache_sequence_length * num_heads * head_size +
                                 static_cast<int64_t>(n) * head_size + h0)
                              : (static_cast<int64_t>(bn) * cache_sequence_length * head_size + h0);
  const int in_row_stride = is_input_bsnh ? (num_heads * head_size) : head_size;

  const int row_step = rows_per_block * gridDim.x;
  for (int s = blockIdx.x * rows_per_block + row_in_block; s < limit; s += row_step) {
    const LoadVec packed = *reinterpret_cast<const LoadVec*>(
        quantized_data + in_base + static_cast<int64_t>(s) * in_row_stride);
    const auto* raw = reinterpret_cast<const T_QUANT*>(&packed);

    alignas(16) T out[kVecSize];
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      float value;
#ifdef USE_FP8_KV_CACHE
      if constexpr (std::is_same<T_QUANT, __nv_fp8_e4m3>::value) {
        value = static_cast<float>(raw[i]);
      } else
#endif
      {
        value = static_cast<float>(reinterpret_cast<const int8_t*>(raw)[i]);
      }
      out[i] = static_cast<T>(value * scales[i]);
    }

    StoreVec* dst = reinterpret_cast<StoreVec*>(dequantized_data + out_base +
                                                static_cast<int64_t>(s) * head_size);
#pragma unroll
    for (int i = 0; i < kStoresPerVec; i++) {
      dst[i] = *reinterpret_cast<const StoreVec*>(&out[i * kElemsPerStore]);
    }
  }
}

template <typename T, typename T_QUANT, typename T_SCALE, int kVecSize>
Status LaunchDequantizeKVVectorized(cudaStream_t stream, T* dequantized_data,
                                    const T_QUANT* quantized_data, const T_SCALE* scale,
                                    const int* valid_seq_lens, int batch_size, int num_heads,
                                    int cache_sequence_length, int head_size,
                                    KVQuantizationType quant_type, bool is_input_bsnh) {
  const int vecs_per_row = head_size / kVecSize;
  const int rows_per_block = kThreadsPerBlock / vecs_per_row;
  const int row_chunks = (cache_sequence_length + rows_per_block - 1) / rows_per_block;

  // Cap the grid so that a short sequence in a large cache does not launch a wave of blocks that
  // immediately exit; the kernel loops over the remaining chunks. The cap only depends on shapes,
  // so the launch configuration stays constant across decode steps (CUDA graph replay safe).
  const int grid_y = batch_size * num_heads;
  const int max_chunks = std::max(1, 1024 / grid_y);
  const dim3 grid(static_cast<unsigned>(std::min(row_chunks, max_chunks)),
                  static_cast<unsigned>(grid_y));

  DequantizeKVVectorizedKernel<T, T_QUANT, T_SCALE, kVecSize><<<grid, kThreadsPerBlock, 0, stream>>>(
      dequantized_data, quantized_data, scale, valid_seq_lens,
      num_heads, cache_sequence_length, head_size, quant_type, is_input_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchDequantizeKV(cudaStream_t stream, T* dequantized_data,
                          const T_QUANT* quantized_data, const T_SCALE* scale,
                          const int* past_seq_lens, int batch_size, int num_heads,
                          int cache_sequence_length,
                          int head_size, int bit_width,
                          KVQuantizationType quant_type,
                          bool is_input_bsnh,
                          const int* valid_seq_lens = nullptr) {
  if (cache_sequence_length == 0) return Status::OK();

  // Fast path: 8-bit caches (INT8 / FP8) whose head size can be split into aligned vectors.
  // past_seq_lens (zero-fill semantics) is only used by callers that do not have valid_seq_lens.
  if constexpr (sizeof(T_QUANT) == 1 && sizeof(T) == 2) {
    if (bit_width == 8 && past_seq_lens == nullptr) {
      if (head_size % 32 == 0) {
        return LaunchDequantizeKVVectorized<T, T_QUANT, T_SCALE, 32>(
            stream, dequantized_data, quantized_data, scale, valid_seq_lens,
            batch_size, num_heads, cache_sequence_length, head_size, quant_type, is_input_bsnh);
      }
      if (head_size % 16 == 0) {
        return LaunchDequantizeKVVectorized<T, T_QUANT, T_SCALE, 16>(
            stream, dequantized_data, quantized_data, scale, valid_seq_lens,
            batch_size, num_heads, cache_sequence_length, head_size, quant_type, is_input_bsnh);
      }

      assert(head_size % 8 == 0);  // GQA has validated head_size that is a multiple of 8 in CheckInputs.
      return LaunchDequantizeKVVectorized<T, T_QUANT, T_SCALE, 8>(
          stream, dequantized_data, quantized_data, scale, valid_seq_lens,
          batch_size, num_heads, cache_sequence_length, head_size, quant_type, is_input_bsnh);
    }
  }

  // Output buffer uses cache_sequence_length stride
  int64_t total_elements = static_cast<int64_t>(batch_size) * num_heads * cache_sequence_length * head_size;
  const int blocks = static_cast<int>((total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
  DequantizeKernel<T, T_QUANT, T_SCALE><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dequantized_data, quantized_data, scale, past_seq_lens, valid_seq_lens,
      batch_size, num_heads, cache_sequence_length,
      head_size, bit_width, quant_type, is_input_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

// ============================================================================
// Folding per-channel KV dequantization scales into Q and into the attention output.
//
// The XQA decode kernel only understands a *scalar* dequantization scale: it folds k_scale into
// qkScale (applied to the Q*K.T accumulator) and v_scale into voScale (applied to the P*V
// accumulator). That is why per-channel quantized caches used to be disqualified from XQA and had
// to fall back to "dequantize the whole cache, then run flash attention", which costs O(context)
// memory traffic on every decode step.
//
// Per-channel scales can be moved out of the kernel exactly, because dequantization is linear and
// the channel index is the *contraction* dim for K and the *free* dim for V:
//
//   scores_t = sum_d q_d * (k_td * sk_d)          = sum_d (q_d * sk_d) * k_td
//   out_d    = sum_t p_t   * (v_td * sv_d)        = (sum_t p_t * v_td) * sv_d
//
// So pre-scaling Q by sk (per kv-head, per channel) and post-scaling the attention output by sv
// produces exactly the same result as dequantizing the cache, with two elementwise passes over
// tensors of shape [batch, seq, num_heads, head_size] -- i.e. O(1) in context length.
//
// p_t is unaffected because the softmax only sees the (already correct) scores, so attention sinks,
// sliding window and the multi-block (Flash Decoding) reduction all stay valid: every step after
// the P*V accumulation is linear in the accumulator.
// ============================================================================
template <typename T>
__global__ void ScaleHeadsByChannelScaleKernel(T* __restrict__ dst,
                                               const T* __restrict__ src,
                                               const float* __restrict__ channel_scale,
                                               int num_heads, int head_size, int group_size,
                                               int64_t total_elements) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total_elements) {
    return;
  }
  const int h = static_cast<int>(i / head_size) % num_heads;  // q head, layout is [..., num_heads, head_size]
  const int d = static_cast<int>(i % head_size);
  const float scale = channel_scale[(h / group_size) * head_size + d];  // scale is [kv_num_heads, head_size]
  dst[i] = static_cast<T>(static_cast<float>(src[i]) * scale);
}

// dst may alias src (the common case: scale the rotated Q in place, or the output in place).
template <typename T>
Status LaunchScaleHeadsByChannelScale(cudaStream_t stream, T* dst, const T* src,
                                      const float* channel_scale, int batch_size, int sequence_length,
                                      int num_heads, int kv_num_heads, int head_size) {
  const int64_t total_elements = static_cast<int64_t>(batch_size) * sequence_length * num_heads * head_size;
  if (total_elements == 0) {
    return Status::OK();
  }
  const int blocks = static_cast<int>((total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
  ScaleHeadsByChannelScaleKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      dst, src, channel_scale, num_heads, head_size, num_heads / kv_num_heads, total_elements);
  return CUDA_CALL(cudaGetLastError());
}

// Quantization Kernel: Converts Floating Point (T) cache to Quantized (Int8/Int4/FP8) values.
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
                               bool is_input_bsnh) {
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

        reinterpret_cast<int8_t*>(quantized_data)[out_idx] = 0;
#ifdef USE_FP8_KV_CACHE
      } else if constexpr (std::is_same<T_QUANT, __nv_fp8_e4m3>::value) {  // FP8
        int64_t out_idx = i;

        reinterpret_cast<__nv_fp8_e4m3*>(quantized_data)[out_idx] = __nv_fp8_e4m3(0.0f);
#endif
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

        // INT4 uses +8 bias, so zero values pack to 0x88
        reinterpret_cast<uint8_t*>(quantized_data)[out_idx] = kInt4ZeroPacked;
#endif
      }
      continue;
    }

    int64_t output_idx = i;

#ifdef USE_FP8_KV_CACHE
    if constexpr (std::is_same<T_QUANT, __nv_fp8_e4m3>::value) {
      int h = h_packed;
      float scale_val = 1.0f;
      if (quant_type == KVQuantizationType::PER_TENSOR) {
        scale_val = static_cast<float>(scale[0]);
      } else {  // PER_CHANNEL
        int scale_idx = n * head_size + h;
        scale_val = static_cast<float>(scale[scale_idx]);
      }

      float inv_scale = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
      int64_t flattened_input_idx = is_input_bsnh ? (static_cast<int64_t>(b) * input_sequence_length * num_heads * head_size +
                                                     static_cast<int64_t>(s) * num_heads * head_size +
                                                     static_cast<int64_t>(n) * head_size +
                                                     h)
                                                  : ((int64_t)b * num_heads * input_sequence_length * head_size +
                                                     (int64_t)n * input_sequence_length * head_size +
                                                     (int64_t)s * head_size +
                                                     h);
      float val_float = static_cast<float>(dequantized_data[flattened_input_idx]) * inv_scale;

      // Clamp to FP8 E4M3 range and convert
      val_float = fmaxf(-kFp8E4M3Max, fminf(kFp8E4M3Max, val_float));
      reinterpret_cast<__nv_fp8_e4m3*>(quantized_data)[output_idx] = __nv_fp8_e4m3(val_float);
    } else
#endif
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
                        bool is_input_bsnh) {
  assert(total_seq_lens != nullptr);
  if (cache_sequence_length == 0) return Status::OK();

  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_packed_elements = batch_size * num_heads * cache_sequence_length * elements_per_head_packed;

  int blocks = (total_packed_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;

  QuantizeKernel<T, T_QUANT, T_SCALE><<<blocks, kThreadsPerBlock, 0, stream>>>(
      quantized_data, dequantized_data, scale, past_seq_lens, total_seq_lens, total_packed_elements,
      input_sequence_length, cache_sequence_length, num_heads, head_size, bit_width, quant_type, is_input_bsnh);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
