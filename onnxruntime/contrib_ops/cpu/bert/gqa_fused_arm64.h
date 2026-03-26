// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Fused flash-attention style GQA implementation optimized for ARM64.
// Key optimizations over the baseline gqa_attention_base.h implementation:
//
// 1. Fused QK+Softmax+AV: Eliminates the large BxNxSxT attention_probs intermediate
//    buffer by using online (streaming) softmax. This dramatically reduces memory
//    bandwidth requirements which is critical for power-constrained ARM devices.
//
// 2. KV-tiled computation: Processes K/V in tiles that fit in L1/L2 cache,
//    improving data reuse and reducing cache misses.
//
// 3. Combined KV cache update: Fuses the KV concat with the attention computation
//    to avoid redundant memory reads of the KV cache.
//
// 4. Optimized parallelization: Uses per-head parallelism without over-sharding
//    for the typical batch_size=1 case.

#include "contrib_ops/cpu/bert/attention_helper.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#include <arm_neon.h>
#endif

// Software prefetch hints
#if defined(__GNUC__) || defined(__clang__)
#define GQA_PREFETCH_L2(addr) __builtin_prefetch(addr, 0, 2)
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
#include <intrin.h>
#define GQA_PREFETCH_L2(addr) __prefetch(addr)
#else
#define GQA_PREFETCH_L2(addr) ((void)0)
#endif

namespace onnxruntime {
namespace contrib {
namespace gqa_fused {

// Tile size for KV sequence dimension. Chosen to keep K tile + V tile + partial outputs
// within L1 cache (~32-48KB on typical ARM cores).
// For head_size=96 and float: tile of 64 KV positions = 64*96*4 = 24KB for K + 24KB for V = 48KB
// For head_size=96 and fp16: tile of 128 KV positions = 128*96*2 = 24KB for K + 24KB for V = 48KB
constexpr size_t kKVTileSizeFloat = 64;
constexpr size_t kKVTileSizeHalf = 128;

// Returns the appropriate KV tile size based on data type and head size.
template <typename T>
inline size_t GetKVTileSize(size_t head_size) {
  // Target: keep K_tile + V_tile + scratch within ~48KB (L1)
  // Each tile: tile_size * head_size * sizeof(T)
  // Two tiles (K+V): 2 * tile_size * head_size * sizeof(T) <= 48KB
  constexpr size_t kL1Budget = 48 * 1024;
  size_t max_tile = kL1Budget / (2 * head_size * sizeof(T));
  // Round down to multiple of 8 for NEON alignment
  max_tile = (max_tile / 8) * 8;
  if (max_tile < 8) max_tile = 8;
  // Cap at reasonable maximum
  if (max_tile > 256) max_tile = 256;
  return max_tile;
}

// Online softmax state for a single row
struct SoftmaxState {
  float max_val;
  float sum_exp;

  SoftmaxState() : max_val(-std::numeric_limits<float>::infinity()), sum_exp(0.0f) {}
};

// Compute dot product of two float vectors of length `len`.
// Uses Neon intrinsics when available for ARM64.
inline float DotProduct(const float* a, const float* b, size_t len) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    float32x4_t va0 = vld1q_f32(a + i);
    float32x4_t vb0 = vld1q_f32(b + i);
    float32x4_t va1 = vld1q_f32(a + i + 4);
    float32x4_t vb1 = vld1q_f32(b + i + 4);
    sum0 = vfmaq_f32(sum0, va0, vb0);
    sum1 = vfmaq_f32(sum1, va1, vb1);
  }
  for (; i + 4 <= len; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum0 = vfmaq_f32(sum0, va, vb);
  }
  sum0 = vaddq_f32(sum0, sum1);
  float result = vaddvq_f32(sum0);
  for (; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
#else
  float result = 0.0f;
  for (size_t i = 0; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

// Scale-and-accumulate: output[j] += scale * v[j] for j in [0, len)
// Uses Neon intrinsics when available.
inline void ScaleAccumulate(float* output, const float* v, float scale, size_t len) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
  float32x4_t vscale = vdupq_n_f32(scale);
  size_t j = 0;
  for (; j + 8 <= len; j += 8) {
    float32x4_t vo0 = vld1q_f32(output + j);
    float32x4_t vv0 = vld1q_f32(v + j);
    float32x4_t vo1 = vld1q_f32(output + j + 4);
    float32x4_t vv1 = vld1q_f32(v + j + 4);
    vo0 = vfmaq_f32(vo0, vv0, vscale);
    vo1 = vfmaq_f32(vo1, vv1, vscale);
    vst1q_f32(output + j, vo0);
    vst1q_f32(output + j + 4, vo1);
  }
  for (; j + 4 <= len; j += 4) {
    float32x4_t vo = vld1q_f32(output + j);
    float32x4_t vv = vld1q_f32(v + j);
    vo = vfmaq_f32(vo, vv, vscale);
    vst1q_f32(output + j, vo);
  }
  for (; j < len; j++) {
    output[j] += scale * v[j];
  }
#else
  for (size_t j = 0; j < len; j++) {
    output[j] += scale * v[j];
  }
#endif
}

// Scale output[j] *= scale for j in [0, len)
inline void ScaleInPlace(float* output, float scale, size_t len) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
  float32x4_t vscale = vdupq_n_f32(scale);
  size_t j = 0;
  for (; j + 8 <= len; j += 8) {
    float32x4_t vo0 = vld1q_f32(output + j);
    float32x4_t vo1 = vld1q_f32(output + j + 4);
    vo0 = vmulq_f32(vo0, vscale);
    vo1 = vmulq_f32(vo1, vscale);
    vst1q_f32(output + j, vo0);
    vst1q_f32(output + j + 4, vo1);
  }
  for (; j + 4 <= len; j += 4) {
    float32x4_t vo = vld1q_f32(output + j);
    vo = vmulq_f32(vo, vscale);
    vst1q_f32(output + j, vo);
  }
  for (; j < len; j++) {
    output[j] *= scale;
  }
#else
  for (size_t j = 0; j < len; j++) {
    output[j] *= scale;
  }
#endif
}

// =====================================================
// FP16 Native NEON Helpers
// =====================================================

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)

// Broadcast a float scalar to float16x8_t via MLFloat16 conversion.
inline float16x8_t BroadcastFp16(float val) {
  MLFloat16 h(val);
  return vreinterpretq_f16_u16(vdupq_n_u16(h.val));
}

// Dot product of two FP16 vectors, returning float for softmax precision.
// Accumulates in FP16 NEON registers (2x throughput over FP32), then
// widens to FP32 for the final horizontal reduction.
inline float DotProductFp16(const MLFloat16* a, const MLFloat16* b, size_t len) {
  const uint16_t* a16 = reinterpret_cast<const uint16_t*>(a);
  const uint16_t* b16 = reinterpret_cast<const uint16_t*>(b);
  float16x8_t sum0 = vreinterpretq_f16_u16(vdupq_n_u16(0));
  float16x8_t sum1 = vreinterpretq_f16_u16(vdupq_n_u16(0));
  size_t i = 0;
  for (; i + 16 <= len; i += 16) {
    float16x8_t va0 = vreinterpretq_f16_u16(vld1q_u16(a16 + i));
    float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(b16 + i));
    float16x8_t va1 = vreinterpretq_f16_u16(vld1q_u16(a16 + i + 8));
    float16x8_t vb1 = vreinterpretq_f16_u16(vld1q_u16(b16 + i + 8));
    sum0 = vfmaq_f16(sum0, va0, vb0);
    sum1 = vfmaq_f16(sum1, va1, vb1);
  }
  for (; i + 8 <= len; i += 8) {
    float16x8_t va = vreinterpretq_f16_u16(vld1q_u16(a16 + i));
    float16x8_t vb = vreinterpretq_f16_u16(vld1q_u16(b16 + i));
    sum0 = vfmaq_f16(sum0, va, vb);
  }
  sum0 = vaddq_f16(sum0, sum1);
  // Widen to FP32 for accurate final reduction
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(sum0));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(sum0));
  float32x4_t s4 = vaddq_f32(lo, hi);
  float result = vaddvq_f32(s4);
  // Scalar remainder
  for (; i < len; i++) {
    result += MLFloat16::FromBits(a16[i]).ToFloat() * MLFloat16::FromBits(b16[i]).ToFloat();
  }
  return result;
}

// FP16 scale-and-accumulate: output[j] += scale * v[j]
// All vectors in FP16, scale is float broadcast to FP16.
inline void ScaleAccumulateFp16(MLFloat16* output, const MLFloat16* v, float scale, size_t len) {
  uint16_t* out16 = reinterpret_cast<uint16_t*>(output);
  const uint16_t* v16 = reinterpret_cast<const uint16_t*>(v);
  float16x8_t vscale = BroadcastFp16(scale);
  size_t j = 0;
  for (; j + 16 <= len; j += 16) {
    float16x8_t vo0 = vreinterpretq_f16_u16(vld1q_u16(out16 + j));
    float16x8_t vv0 = vreinterpretq_f16_u16(vld1q_u16(v16 + j));
    float16x8_t vo1 = vreinterpretq_f16_u16(vld1q_u16(out16 + j + 8));
    float16x8_t vv1 = vreinterpretq_f16_u16(vld1q_u16(v16 + j + 8));
    vo0 = vfmaq_f16(vo0, vv0, vscale);
    vo1 = vfmaq_f16(vo1, vv1, vscale);
    vst1q_u16(out16 + j, vreinterpretq_u16_f16(vo0));
    vst1q_u16(out16 + j + 8, vreinterpretq_u16_f16(vo1));
  }
  for (; j + 8 <= len; j += 8) {
    float16x8_t vo = vreinterpretq_f16_u16(vld1q_u16(out16 + j));
    float16x8_t vv = vreinterpretq_f16_u16(vld1q_u16(v16 + j));
    vo = vfmaq_f16(vo, vv, vscale);
    vst1q_u16(out16 + j, vreinterpretq_u16_f16(vo));
  }
  for (; j < len; j++) {
    float val = MLFloat16::FromBits(out16[j]).ToFloat() + scale * MLFloat16::FromBits(v16[j]).ToFloat();
    out16[j] = MLFloat16(val).val;
  }
}

// FP16 in-place scale: output[j] *= scale
// Vector in FP16, scale is float broadcast to FP16.
inline void ScaleInPlaceFp16(MLFloat16* output, float scale, size_t len) {
  uint16_t* out16 = reinterpret_cast<uint16_t*>(output);
  float16x8_t vscale = BroadcastFp16(scale);
  size_t j = 0;
  for (; j + 16 <= len; j += 16) {
    float16x8_t vo0 = vreinterpretq_f16_u16(vld1q_u16(out16 + j));
    float16x8_t vo1 = vreinterpretq_f16_u16(vld1q_u16(out16 + j + 8));
    vo0 = vmulq_f16(vo0, vscale);
    vo1 = vmulq_f16(vo1, vscale);
    vst1q_u16(out16 + j, vreinterpretq_u16_f16(vo0));
    vst1q_u16(out16 + j + 8, vreinterpretq_u16_f16(vo1));
  }
  for (; j + 8 <= len; j += 8) {
    float16x8_t vo = vreinterpretq_f16_u16(vld1q_u16(out16 + j));
    vo = vmulq_f16(vo, vscale);
    vst1q_u16(out16 + j, vreinterpretq_u16_f16(vo));
  }
  for (; j < len; j++) {
    float val = MLFloat16::FromBits(out16[j]).ToFloat() * scale;
    out16[j] = MLFloat16(val).val;
  }
}

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

// Fused QK*softmax*V computation for a single query head using online softmax.
//
// This implements the Flash-Attention style algorithm:
//   For each tile of K/V positions:
//     1. Compute QK scores for this tile (dot products)
//     2. Apply causal mask
//     3. Update running max and sum for online softmax
//     4. Rescale previous accumulated output
//     5. Accumulate attention_weight * V_tile into output
//
// Parameters:
//   q:              pointer to query vector(s) for this head, shape [S, H]
//   k:              pointer to key cache for this KV head, shape [T, H]
//   v:              pointer to value cache for this KV head, shape [T, H]
//   output:         output buffer, shape [S, H] (accumulator, in FP32)
//   head_size:      dimension of each head (H)
//   sequence_length: number of query positions (S)
//   total_seqlen:   total KV sequence length (T)
//   past_seqlen:    number of past positions (for causal mask offset)
//   present_buffer_sequence_length: stride for present KV buffer
//   alpha:          scale factor (1/sqrt(head_size))
//   softcap:        softcap value (0 = disabled)
//   local_window_size: local window size (-1 = disabled)
//   use_smooth_softmax: whether to use smooth softmax
//   head_sink_val:  head sink value for smooth softmax (0 if not used)
//   attention_bias: optional attention bias pointer for this head [S, T_bias]
//   attention_bias_stride: stride for attention_bias (T dimension of bias shape)
//   qk_scores_buf:  scratch buffer for QK scores, size >= kv_tile_size
//   output_qk:      optional QK output buffer
//   output_qk_stride: stride for output_qk (total_sequence_length)
//   qk_output_type: QKOutputType enum value
template <typename T>
void FusedAttentionHead(
    const T* q,
    const T* k,
    const T* v,
    float* output,  // [S, head_size] - FP32 accumulator
    size_t head_size,
    size_t sequence_length,
    size_t total_seqlen,
    size_t past_seqlen,
    size_t present_buffer_sequence_length,
    float alpha,
    float softcap,
    int local_window_size,
    bool use_smooth_softmax,
    float head_sink_val,
    const T* attention_bias,
    ptrdiff_t attention_bias_stride,
    float* qk_scores_buf,  // scratch [kv_tile_size]
    float* q_fp32_buf,     // scratch [S * H] for FP16->FP32 Q conversion (nullptr for float)
    float* kv_tile_fp32_buf,  // scratch [2 * kv_tile_size * H] for FP16->FP32 K/V tiles (nullptr for float)
    T* output_qk,
    size_t output_qk_stride,
    int qk_output_type) {
  const size_t kv_tile_size = GetKVTileSize<T>(head_size);

  // Per-row online softmax state (small, stack-friendly for typical S values)
  SoftmaxState softmax_states_stack[64];
  std::vector<SoftmaxState> softmax_states_heap;
  SoftmaxState* softmax_states;
  if (sequence_length <= 64) {
    softmax_states = softmax_states_stack;
    for (size_t s = 0; s < sequence_length; s++) {
      softmax_states[s] = SoftmaxState();
    }
  } else {
    softmax_states_heap.resize(sequence_length);
    softmax_states = softmax_states_heap.data();
  }

  // Zero output accumulator
  memset(output, 0, sequence_length * head_size * sizeof(float));

  // FP32 Q data: use directly for float, convert for FP16
  const float* q_fp32 = nullptr;
  if constexpr (std::is_same<T, float>::value) {
    q_fp32 = q;
  } else {
    MlasConvertHalfToFloatBuffer(q, q_fp32_buf, sequence_length * head_size);
    q_fp32 = q_fp32_buf;
  }

  // Scratch pointers for K/V tile FP32 conversion
  float* k_tile_fp32 = kv_tile_fp32_buf;
  float* v_tile_fp32 = kv_tile_fp32_buf != nullptr
                            ? kv_tile_fp32_buf + kv_tile_size * head_size
                            : nullptr;

  // Process K/V in tiles along the sequence dimension
  for (size_t kv_start = 0; kv_start < total_seqlen; kv_start += kv_tile_size) {
    const size_t kv_end = std::min(kv_start + kv_tile_size, total_seqlen);
    const size_t tile_len = kv_end - kv_start;

    // Prefetch next K/V tile into L2 cache
    if (kv_end < total_seqlen) {
      GQA_PREFETCH_L2(reinterpret_cast<const char*>(k + kv_end * head_size));
      GQA_PREFETCH_L2(reinterpret_cast<const char*>(v + kv_end * head_size));
    }

    // Get FP32 pointers to K and V tiles
    const float* k_tile;
    const float* v_tile;
    size_t k_stride = head_size;
    size_t v_stride = head_size;

    if constexpr (std::is_same<T, float>::value) {
      k_tile = k + kv_start * head_size;
      v_tile = v + kv_start * head_size;
    } else {
      // Convert K tile to FP32
      MlasConvertHalfToFloatBuffer(k + kv_start * head_size, k_tile_fp32, tile_len * head_size);
      k_tile = k_tile_fp32;
      // Convert V tile to FP32
      MlasConvertHalfToFloatBuffer(v + kv_start * head_size, v_tile_fp32, tile_len * head_size);
      v_tile = v_tile_fp32;
    }

    // For each query position
    for (size_t s = 0; s < sequence_length; s++) {
      const size_t seq_causal_length = past_seqlen + s + 1;
      const float* q_row = q_fp32 + s * head_size;
      float* out_row = output + s * head_size;
      SoftmaxState& state = softmax_states[s];

      // Determine the effective range of this tile that is causally visible
      if (kv_start >= seq_causal_length) {
        // Entire tile is masked out (future positions)
        continue;
      }

      const size_t visible_end = std::min(tile_len, seq_causal_length - kv_start);

      // Determine local window bounds
      size_t local_start = 0;
      if (local_window_size >= 0 && seq_causal_length > static_cast<size_t>(local_window_size)) {
        size_t window_start = seq_causal_length - static_cast<size_t>(local_window_size);
        if (window_start > kv_start) {
          local_start = window_start - kv_start;
        }
        if (local_start >= visible_end) {
          // Entire visible portion is before the local window
          continue;
        }
      }

      // Compute QK scores for visible positions in this tile
      float tile_max = -std::numeric_limits<float>::infinity();
      for (size_t t = local_start; t < visible_end; t++) {
        float score = DotProduct(q_row, k_tile + t * k_stride, head_size) * alpha;

        // Apply softcap
        if (softcap > 0.0f) {
          score = softcap * std::tanh(score / softcap);
        }

        // Apply attention bias
        if (attention_bias != nullptr) {
          size_t bias_pos = kv_start + t;
          if constexpr (std::is_same<T, float>::value) {
            score += attention_bias[s * attention_bias_stride + bias_pos];
          } else {
            float bias_val;
            MlasConvertHalfToFloatBuffer(&attention_bias[s * attention_bias_stride + bias_pos], &bias_val, 1);
            score += bias_val;
          }
        }

        qk_scores_buf[t - local_start] = score;
        if (score > tile_max) {
          tile_max = score;
        }
      }

      const size_t num_scores = visible_end - local_start;
      if (num_scores == 0) continue;

      // Write QK output before softmax if requested
      if (output_qk != nullptr && qk_output_type == 1) {
        // BEFORE_SOFTMAX: write raw scores
        // We need to reconstruct the full row including zeros for masked positions
        // This is a debug output, so we just write the visible part
        T* qk_row = output_qk + s * output_qk_stride;
        for (size_t t = 0; t < local_start; t++) {
          if constexpr (std::is_same<T, float>::value) {
            qk_row[kv_start + t] = 0.0f;
          } else {
            qk_row[kv_start + t] = MLFloat16(0.0f);
          }
        }
        for (size_t t = local_start; t < visible_end; t++) {
          if constexpr (std::is_same<T, float>::value) {
            qk_row[kv_start + t] = qk_scores_buf[t - local_start];
          } else {
            qk_row[kv_start + t] = MLFloat16(qk_scores_buf[t - local_start]);
          }
        }
        for (size_t t = visible_end; t < tile_len; t++) {
          if constexpr (std::is_same<T, float>::value) {
            qk_row[kv_start + t] = 0.0f;
          } else {
            qk_row[kv_start + t] = MLFloat16(0.0f);
          }
        }
      }

      // Online softmax update:
      // old_max = state.max_val
      // new_max = max(old_max, tile_max)
      // correction = exp(old_max - new_max)
      // state.sum_exp = state.sum_exp * correction + sum(exp(score_i - new_max))
      // output = output * correction + sum(exp(score_i - new_max) * v_i)

      float old_max = state.max_val;
      float new_max = std::max(old_max, tile_max);

      // Correction factor for previously accumulated values
      float correction = 1.0f;
      if (state.sum_exp > 0.0f) {
        correction = std::exp(old_max - new_max);
        // Rescale previous output accumulation
        ScaleInPlace(out_row, correction, head_size);
        state.sum_exp *= correction;
      }
      state.max_val = new_max;

      // Compute exp(score - new_max) and accumulate V
      for (size_t t = 0; t < num_scores; t++) {
        float exp_score = std::exp(qk_scores_buf[t] - new_max);
        state.sum_exp += exp_score;

        // Accumulate weighted V
        const float* v_row = v_tile + (local_start + t) * v_stride;
        ScaleAccumulate(out_row, v_row, exp_score, head_size);
      }
    }
  }

  // Final normalization: output /= sum_exp
  // Also handle smooth softmax by adding exp(sink - max) to denominator
  for (size_t s = 0; s < sequence_length; s++) {
    SoftmaxState& state = softmax_states[s];
    float denom = state.sum_exp;

    if (use_smooth_softmax) {
      // Add exp(sink - max) to denominator
      denom += std::exp(head_sink_val - state.max_val);
    }

    if (denom > 0.0f) {
      float inv_denom = 1.0f / denom;
      ScaleInPlace(output + s * head_size, inv_denom, head_size);
    }
  }

  // Write QK output after softmax if requested
  // Note: This is approximate since we don't store the full attention matrix.
  // For debug purposes with the fused path, we'd need a separate non-fused run.
  // The qk_output_type == AFTER_SOFTMAX case is inherently incompatible with fused attention.
}

// =====================================================
// FP16 Native Fused Attention Head
// =====================================================
//
// Fully native FP16 path: reads Q/K/V as FP16, accumulates output in FP16,
// with only softmax state (max, sum_exp) and exp() in FP32.
// Benefits over FP32-conversion path:
//   - 2x NEON throughput (float16x8_t = 8 lanes vs float32x4_t = 4 lanes)
//   - 2x memory bandwidth (no FP16->FP32 conversion of K/V tiles)
//   - 2x effective KV tile size for the same L1 cache budget
//   - Eliminates ~50KB/thread of FP32 scratch buffers

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)

inline void FusedAttentionHeadFp16(
    const MLFloat16* q,
    const MLFloat16* k,
    const MLFloat16* v,
    MLFloat16* output,  // [S, head_size] - FP16 accumulator
    size_t head_size,
    size_t sequence_length,
    size_t total_seqlen,
    size_t past_seqlen,
    float alpha,
    float softcap,
    int local_window_size,
    bool use_smooth_softmax,
    float head_sink_val,
    const MLFloat16* attention_bias,
    ptrdiff_t attention_bias_stride,
    float* qk_scores_buf,
    MLFloat16* output_qk,
    size_t output_qk_stride,
    int qk_output_type) {
  const size_t kv_tile_size = GetKVTileSize<MLFloat16>(head_size);

  SoftmaxState softmax_states_stack[64];
  std::vector<SoftmaxState> softmax_states_heap;
  SoftmaxState* softmax_states;
  if (sequence_length <= 64) {
    softmax_states = softmax_states_stack;
    for (size_t s = 0; s < sequence_length; s++) {
      softmax_states[s] = SoftmaxState();
    }
  } else {
    softmax_states_heap.resize(sequence_length);
    softmax_states = softmax_states_heap.data();
  }

  // Zero FP16 output accumulator
  memset(output, 0, sequence_length * head_size * sizeof(MLFloat16));

  // Process KV in tiles (tile is 2x larger than FP32 path for same L1 budget)
  for (size_t kv_start = 0; kv_start < total_seqlen; kv_start += kv_tile_size) {
    const size_t kv_end = std::min(kv_start + kv_tile_size, total_seqlen);
    const size_t tile_len = kv_end - kv_start;

    // Prefetch next K/V tile
    if (kv_end < total_seqlen) {
      GQA_PREFETCH_L2(reinterpret_cast<const char*>(k + kv_end * head_size));
      GQA_PREFETCH_L2(reinterpret_cast<const char*>(v + kv_end * head_size));
    }

    // Read K/V directly as FP16 - no conversion needed
    const MLFloat16* k_tile = k + kv_start * head_size;
    const MLFloat16* v_tile = v + kv_start * head_size;

    for (size_t s = 0; s < sequence_length; s++) {
      const size_t seq_causal_length = past_seqlen + s + 1;
      const MLFloat16* q_row = q + s * head_size;
      MLFloat16* out_row = output + s * head_size;
      SoftmaxState& state = softmax_states[s];

      if (kv_start >= seq_causal_length) continue;

      const size_t visible_end = std::min(tile_len, seq_causal_length - kv_start);

      size_t local_start = 0;
      if (local_window_size >= 0 && seq_causal_length > static_cast<size_t>(local_window_size)) {
        size_t window_start = seq_causal_length - static_cast<size_t>(local_window_size);
        if (window_start > kv_start) {
          local_start = window_start - kv_start;
        }
        if (local_start >= visible_end) continue;
      }

      // Compute QK scores: native FP16 dot product, result in FP32 for softmax
      float tile_max = -std::numeric_limits<float>::infinity();
      for (size_t t = local_start; t < visible_end; t++) {
        float score = DotProductFp16(q_row, k_tile + t * head_size, head_size) * alpha;

        if (softcap > 0.0f) {
          score = softcap * std::tanh(score / softcap);
        }

        if (attention_bias != nullptr) {
          size_t bias_pos = kv_start + t;
          score += attention_bias[s * attention_bias_stride + bias_pos].ToFloat();
        }

        qk_scores_buf[t - local_start] = score;
        if (score > tile_max) tile_max = score;
      }

      const size_t num_scores = visible_end - local_start;
      if (num_scores == 0) continue;

      // Write QK output before softmax if requested
      if (output_qk != nullptr && qk_output_type == 1) {
        MLFloat16* qk_row = output_qk + s * output_qk_stride;
        for (size_t t = 0; t < local_start; t++) {
          qk_row[kv_start + t] = MLFloat16(0.0f);
        }
        for (size_t t = local_start; t < visible_end; t++) {
          qk_row[kv_start + t] = MLFloat16(qk_scores_buf[t - local_start]);
        }
        for (size_t t = visible_end; t < tile_len; t++) {
          qk_row[kv_start + t] = MLFloat16(0.0f);
        }
      }

      // Online softmax update (scalars in FP32, vector ops in FP16)
      float old_max = state.max_val;
      float new_max = std::max(old_max, tile_max);

      if (state.sum_exp > 0.0f) {
        float correction = std::exp(old_max - new_max);
        ScaleInPlaceFp16(out_row, correction, head_size);
        state.sum_exp *= correction;
      }
      state.max_val = new_max;

      // Accumulate weighted V in FP16
      for (size_t t = 0; t < num_scores; t++) {
        float exp_score = std::exp(qk_scores_buf[t] - new_max);
        state.sum_exp += exp_score;
        const MLFloat16* v_row = v_tile + (local_start + t) * head_size;
        ScaleAccumulateFp16(out_row, v_row, exp_score, head_size);
      }
    }
  }

  // Final normalization
  for (size_t s = 0; s < sequence_length; s++) {
    SoftmaxState& state = softmax_states[s];
    float denom = state.sum_exp;
    if (use_smooth_softmax) {
      denom += std::exp(head_sink_val - state.max_val);
    }
    if (denom > 0.0f) {
      ScaleInPlaceFp16(output + s * head_size, 1.0f / denom, head_size);
    }
  }
}

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

// Fused attention for all heads: parallelizes over (batch, num_heads).
// This replaces both ComputeAttentionProbs + ComputeVxAttentionScore.
template <typename T>
void FusedGQAAttention(
    const T* Q,                                // Q data, shape [B, N, S, H]
    const T* K,                                // K new data, shape [B, N_kv, S, H]
    const T* V,                                // V new data, shape [B, N_kv, S, H]
    T* output,                                 // output, shape [B, S, N, H]
    const T* past_key,
    T* present_key,
    const T* past_value,
    T* present_value,
    const int32_t* seqlens_k,
    const T* head_sink,
    const T* attention_bias,
    const gsl::span<const int64_t> attention_bias_shape,
    T* output_qk,
    int qk_output_type,
    size_t batch_size,
    size_t sequence_length,
    size_t total_sequence_length,
    size_t head_size,
    size_t hidden_size,
    int num_heads,
    int kv_num_heads,
    size_t past_buffer_sequence_length,
    size_t present_buffer_sequence_length,
    float alpha,
    float softcap,
    int local_window_size,
    bool use_smooth_softmax,
    bool past_present_share_buffer,
    bool packed_qkv,
    bool is_prompt,
    ThreadPool* tp,
    AllocatorPtr allocator) {
  const size_t kv_num_heads_factor = num_heads / kv_num_heads;
  const size_t q_input_chunk_length = sequence_length * head_size;
  const size_t kv_input_chunk_length = sequence_length * head_size;
  const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;
  const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;

  const ptrdiff_t packed_batch_stride =
      packed_qkv ? SafeInt<ptrdiff_t>(num_heads + 2 * kv_num_heads) * sequence_length * head_size
                 : SafeInt<ptrdiff_t>(0);

  // Zero present KV buffers if not sharing with past
  if (!past_present_share_buffer) {
    if (present_key != nullptr) {
      memset(present_key, 0,
             batch_size * kv_num_heads * present_buffer_sequence_length * head_size * sizeof(T));
    }
    if (present_value != nullptr) {
      memset(present_value, 0,
             batch_size * kv_num_heads * present_buffer_sequence_length * head_size * sizeof(T));
    }
  }

  // Phase 1: Concatenate KV caches (parallelize over KV heads, not Q heads)
  // This ensures each KV head is written exactly once before any Q head reads it.
  if (present_key != nullptr || present_value != nullptr) {
    const size_t kv_loop_len = batch_size * kv_num_heads;

    TensorOpCost kv_cost;
    kv_cost.bytes_loaded = static_cast<double>(present_buff_chunk_length * sizeof(T));
    kv_cost.bytes_stored = static_cast<double>(present_buff_chunk_length * sizeof(T));
    kv_cost.compute_cycles = static_cast<double>(present_buff_chunk_length);

    ThreadPool::TryParallelFor(tp, kv_loop_len, kv_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t kv_i = begin; kv_i != end; ++kv_i) {
        const size_t batch_index = kv_i / kv_num_heads;
        const size_t kv_head_index = kv_i % kv_num_heads;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;
        const size_t past_chunk_length = past_seqlen * head_size;

        if (present_key != nullptr) {
          const T* k_new;
          if (packed_qkv) {
            k_new = K + packed_batch_stride * batch_index + kv_input_chunk_length * kv_head_index;
          } else {
            k_new = K + kv_input_chunk_length * kv_i;
          }
          ConcatStateChunkGQA(past_key, k_new, present_key,
                              present_buff_chunk_length, past_buff_chunk_length,
                              past_chunk_length, kv_input_chunk_length,
                              past_present_share_buffer, kv_i);
        }

        if (present_value != nullptr) {
          const T* v_new;
          if (packed_qkv) {
            v_new = V + packed_batch_stride * batch_index + kv_input_chunk_length * kv_head_index;
          } else {
            v_new = V + kv_input_chunk_length * kv_i;
          }
          ConcatStateChunkGQA(past_value, v_new, present_value,
                              present_buff_chunk_length, past_buff_chunk_length,
                              past_chunk_length, kv_input_chunk_length,
                              past_present_share_buffer, kv_i);
        }
      }
    });
  }

  // Phase 2: Fused QK+Softmax+AV computation (parallelize over Q heads)
  const size_t loop_len = batch_size * num_heads;

  // Cost model: for each head iteration we do O(S * T * H) work
  TensorOpCost unit_cost;
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
  unit_cost.bytes_loaded =
      static_cast<double>((sequence_length + present_buffer_sequence_length) * head_size * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(sequence_length * head_size * sizeof(T));

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
  // Native FP16 path: no FP32 conversion buffers needed.
  // Only requires a small QK scores buffer (FP32 for softmax) and FP16 accumulator.
  if constexpr (std::is_same<T, MLFloat16>::value) {
    const size_t kv_tile_size = GetKVTileSize<MLFloat16>(head_size);

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      std::vector<float> qk_scores_buf(kv_tile_size);
      std::vector<MLFloat16> output_fp16(sequence_length * head_size);

      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads;
        const size_t head_index = i % num_heads;
        const size_t kv_head_index = head_index / kv_num_heads_factor;
        const size_t kv_head_linear = batch_index * kv_num_heads + kv_head_index;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;

        const T* q;
        if (packed_qkv) {
          q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q = Q + q_input_chunk_length * i;
        }

        const T* k_present = present_key != nullptr
                                 ? present_key + kv_head_linear * present_buff_chunk_length
                                 : nullptr;
        const T* v_present = present_value != nullptr
                                 ? present_value + kv_head_linear * present_buff_chunk_length
                                 : nullptr;

        const T* attention_bias_head = nullptr;
        ptrdiff_t attention_bias_stride_val = 0;
        if (attention_bias != nullptr) {
          ptrdiff_t attention_bias_offset = 0;
          attention_bias_stride_val = static_cast<ptrdiff_t>(attention_bias_shape[3]);
          const ptrdiff_t attention_matrix_size = sequence_length * attention_bias_stride_val;
          if (attention_bias_shape[0] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attention_matrix_size;
          }
          if (attention_bias_shape[1] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(head_index) * attention_matrix_size;
          }
          attention_bias_head = attention_bias + attention_bias_offset;
        }

        T* output_qk_head = nullptr;
        size_t output_qk_stride_val = 0;
        if (output_qk != nullptr) {
          output_qk_stride_val = total_sequence_length;
          ptrdiff_t qk_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length *
                                    (batch_index * num_heads + head_index);
          output_qk_head = output_qk + qk_offset;
        }

        float head_sink_val = 0.0f;
        if (head_sink != nullptr) {
          head_sink_val = head_sink[head_index].ToFloat();
        }

        FusedAttentionHeadFp16(
            q, k_present, v_present,
            output_fp16.data(),
            head_size, sequence_length, total_seqlen, past_seqlen,
            alpha, softcap, local_window_size,
            use_smooth_softmax || (head_sink != nullptr),
            head_sink_val,
            attention_bias_head, attention_bias_stride_val,
            qk_scores_buf.data(),
            output_qk_head, output_qk_stride_val, qk_output_type);

        // Write FP16 output directly (no conversion needed)
        T* output_head = output + (batch_index * sequence_length * num_heads + head_index) * head_size;
        for (size_t s = 0; s < sequence_length; s++) {
          memcpy(output_head + s * hidden_size, output_fp16.data() + s * head_size, head_size * sizeof(T));
        }
      }
    });
  } else
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
  {
    // FP32 path (for float type, or FP16 without native NEON support)
    const size_t kv_tile_size = GetKVTileSize<T>(head_size);
    const size_t qk_scratch_size = kv_tile_size;
    const size_t q_fp32_scratch_size = std::is_same<T, float>::value ? 0 : sequence_length * head_size;
    const size_t kv_fp32_scratch_size = std::is_same<T, float>::value ? 0 : 2 * kv_tile_size * head_size;

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      std::vector<float> qk_scores_buf(qk_scratch_size);
      std::vector<float> output_fp32(sequence_length * head_size);
      std::vector<float> q_fp32_buf(q_fp32_scratch_size);
      std::vector<float> kv_tile_fp32_buf(kv_fp32_scratch_size);

      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads;
        const size_t head_index = i % num_heads;
        const size_t kv_head_index = head_index / kv_num_heads_factor;
        const size_t kv_head_linear = batch_index * kv_num_heads + kv_head_index;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;

        const T* q;
        if (packed_qkv) {
          q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q = Q + q_input_chunk_length * i;
        }

        const T* k_present = present_key != nullptr
                                 ? present_key + kv_head_linear * present_buff_chunk_length
                                 : nullptr;
        const T* v_present = present_value != nullptr
                                 ? present_value + kv_head_linear * present_buff_chunk_length
                                 : nullptr;

        const T* attention_bias_head = nullptr;
        ptrdiff_t attention_bias_stride_val = 0;
        if (attention_bias != nullptr) {
          ptrdiff_t attention_bias_offset = 0;
          attention_bias_stride_val = static_cast<ptrdiff_t>(attention_bias_shape[3]);
          const ptrdiff_t attention_matrix_size = sequence_length * attention_bias_stride_val;
          if (attention_bias_shape[0] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attention_matrix_size;
          }
          if (attention_bias_shape[1] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(head_index) * attention_matrix_size;
          }
          attention_bias_head = attention_bias + attention_bias_offset;
        }

        T* output_qk_head = nullptr;
        size_t output_qk_stride_val = 0;
        if (output_qk != nullptr) {
          output_qk_stride_val = total_sequence_length;
          ptrdiff_t qk_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length *
                                    (batch_index * num_heads + head_index);
          output_qk_head = output_qk + qk_offset;
        }

        float head_sink_val = 0.0f;
        if (head_sink != nullptr) {
          if constexpr (std::is_same<T, float>::value) {
            head_sink_val = head_sink[head_index];
          } else {
            MlasConvertHalfToFloatBuffer(&head_sink[head_index], &head_sink_val, 1);
          }
        }

        FusedAttentionHead(
            q, k_present, v_present,
            output_fp32.data(),
            head_size, sequence_length, total_seqlen, past_seqlen,
            present_buffer_sequence_length,
            alpha, softcap, local_window_size,
            use_smooth_softmax || (head_sink != nullptr),
            head_sink_val,
            attention_bias_head, attention_bias_stride_val,
            qk_scores_buf.data(),
            q_fp32_scratch_size > 0 ? q_fp32_buf.data() : nullptr,
            kv_fp32_scratch_size > 0 ? kv_tile_fp32_buf.data() : nullptr,
            output_qk_head, output_qk_stride_val, qk_output_type);

        T* output_head = output + (batch_index * sequence_length * num_heads + head_index) * head_size;
        for (size_t s = 0; s < sequence_length; s++) {
          if constexpr (std::is_same<T, float>::value) {
            memcpy(output_head + s * hidden_size, output_fp32.data() + s * head_size, head_size * sizeof(float));
          } else {
            MlasConvertFloatToHalfBuffer(output_fp32.data() + s * head_size,
                                         output_head + s * hidden_size,
                                         head_size);
          }
        }
      }
    });
  }
}

// Check if fused attention path should be used.
// The fused path is beneficial when:
// 1. The attention_probs buffer would be large (saves memory bandwidth)
// 2. Running on ARM64 where memory bandwidth is constrained
// 3. QK debug output of type AFTER_SOFTMAX is not requested (incompatible with fused path)
inline bool ShouldUseFusedAttention(
    size_t batch_size,
    size_t sequence_length,
    size_t total_sequence_length,
    size_t present_buffer_sequence_length,
    int qk_output_type) {
  // The fused path cannot produce AFTER_SOFTMAX QK output since it doesn't
  // materialize the full attention probability matrix.
  if (qk_output_type == static_cast<int>(QKOutputType::AFTER_SOFTMAX)) {
    return false;
  }

  // Calculate memory savings from avoiding attention_probs allocation.
  // attention_probs size = batch * num_heads * seq_len * present_buf_seq_len * sizeof(T)
  // For batch=1, num_heads=32, seq=6, present=1500, fp32:
  //   = 1 * 32 * 6 * 1500 * 4 = 1.15 MB
  // For prefill: batch=1, num_heads=32, seq=64, present=1000, fp32:
  //   = 1 * 32 * 64 * 1000 * 4 = 8.19 MB
  // The fused path avoids this entirely, keeping data in L1 cache tiles.

  // Use fused path when total KV sequence length is large enough to benefit
  // from tiled processing (overhead of online softmax is amortized).
  const size_t attention_matrix_size = sequence_length * present_buffer_sequence_length;

  // For very small attention matrices, the overhead of online softmax
  // tracking isn't worth it. Threshold: ~256 elements per row.
  if (present_buffer_sequence_length < 32) {
    return false;
  }

  return true;
}

}  // namespace gqa_fused
}  // namespace contrib
}  // namespace onnxruntime
