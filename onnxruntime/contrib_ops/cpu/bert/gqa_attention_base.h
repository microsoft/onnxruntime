// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ARM NEON FP16 intrinsics for flash attention inner loop
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <arm_neon.h>
#endif
#endif

namespace onnxruntime {
namespace contrib {

class GQAAttentionBase {
 protected:
  GQAAttentionBase(const OpKernelInfo& info, bool has_local) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
    kv_num_heads_ = static_cast<int>(kv_num_heads);

    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);

    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

    use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

    local_window_size_ = has_local ? static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1)) : -1;

    qk_output_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)));
  }

  int num_heads_;     // number of attention heads of Q
  int kv_num_heads_;  // number of attention heads of K or V
  float scale_;       // the scaling factor applied before softmax
  float softcap_;
  bool do_rotary_;  // whether or not to use rotary embeddings
  bool rotary_interleaved_;
  int local_window_size_;
  int qk_output_;

  bool use_smooth_softmax_;

  template <typename T>
  Status ApplyAttention(const T* Q,                                 // Q data with shape BxNxSxH
                        const T* K,                                 // K data with shape BxN_kvxSxH
                        const T* V,                                 // V data with shape BxN_kvxSxH
                        const T* head_sink,                         // Head sink for smooth softmax, nullptr if not used
                        const Tensor* attention_bias,               // Attention bias to add to QxK'
                        const Tensor* past_key,                     // past K input tensor (if not using past state)
                        const Tensor* past_value,                   // past V input tensor (if not using past state)
                        Tensor* output,                             // output tensor
                        Tensor* present_key,                        // present K output tensor (if separating present KV)
                        Tensor* present_value,                      // present V output tensor (if separating present KV)
                        Tensor* output_qk,                          // output QK buffer
                        const Tensor* seqlens_k,                    // past sequence lengths tensor
                        GroupQueryAttentionParameters& parameters,  // attention parameters
                        AllocatorPtr allocator,                     // allocator for temporary tensors
                        OpKernelContext* context) const {
    const bool is_prompt = parameters.is_first_prompt;
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int total_sequence_length = parameters.total_sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = static_cast<int>(present_key->Shape().GetDims()[2]);

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    const T* attention_bias_data = attention_bias != nullptr ? attention_bias->Data<T>() : nullptr;
    auto attention_bias_shape = attention_bias != nullptr ? attention_bias->Shape().GetDims() : gsl::span<const int64_t>{};

    bool past_present_share_buffer = past_key_data == present_key_data && past_value_data == present_value_data;

    const T* k = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;
    const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;

    T* output_qk_buffer = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

    if (output_qk_buffer == nullptr) {
      // Flash-attention path: tiled online softmax eliminates the S×T attention_probs buffer
      // and KV pre-packing per GQA group avoids redundant FP16→FP32 conversions.
      ComputeFlashAttention<T>(Q, k, v, head_sink, seqlens_k->Data<int32_t>(),
                               attention_bias_data, output->MutableData<T>(),
                               batch_size, sequence_length, attention_bias_shape,
                               seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, hidden_size,
                               past_key_data, present_key_data, past_value_data, present_value_data,
                               past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);
    } else {
      // Fallback path: full attention_probs materialization needed for QK debug output
      bool gqa_mlas_supported = MlasGQASupported<T>(CblasNoTrans, CblasTrans) &&
                                MlasGQASupported<T>(CblasNoTrans, CblasNoTrans);
      size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * seqlen_present_kv_cache *
                     (gqa_mlas_supported ? sizeof(T) : sizeof(float));
      auto attention_probs = allocator->Alloc(bytes);
      BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

      if (gqa_mlas_supported) {
        ComputeFusedAttention(static_cast<T*>(attention_probs), Q, k, v, head_sink, seqlens_k->Data<int32_t>(),
                              attention_bias_data, output->MutableData<T>(), output_qk_buffer,
                              batch_size, sequence_length, total_sequence_length, attention_bias_shape,
                              seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, hidden_size,
                              past_key_data, present_key_data, past_value_data, present_value_data,
                              past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);
      } else {
        ComputeFusedAttention(static_cast<float*>(attention_probs), Q, k, v, head_sink, seqlens_k->Data<int32_t>(),
                              attention_bias_data, output->MutableData<T>(), output_qk_buffer,
                              batch_size, sequence_length, total_sequence_length, attention_bias_shape,
                              seqlen_past_kv_cache, seqlen_present_kv_cache, head_size, hidden_size,
                              past_key_data, present_key_data, past_value_data, present_value_data,
                              past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);
      }
    }

    return Status::OK();
  }

 private:
  // Flash-attention with tiled online softmax and KV pre-packing per GQA group.
  //
  // Key optimizations:
  //   1. Eliminates the S×T attention_probs buffer entirely using online softmax.
  //   2. Parallel loop over (batch × kv_num_heads): K/V concat + conversion done once per KV head.
  //   3. Tiled access pattern keeps working set in L1 cache.
  //   4. When T=MLFloat16 on ARM64: uses NEON FP16 intrinsics for Q·K dot products (2× throughput),
  //      reads K/V directly in FP16 (no bulk conversion), FP32 accumulators for numerical stability.
  template <typename T>
  void ComputeFlashAttention(const T* Q,
                             const T* K,
                             const T* V,
                             const T* head_sink,
                             const int32_t* seqlens_k,
                             const T* attention_bias,
                             T* output,
                             const size_t batch_size,
                             const size_t sequence_length,
                             const gsl::span<const int64_t> attention_bias_shape,
                             const size_t past_buffer_sequence_length,
                             const size_t present_buffer_sequence_length,
                             const size_t head_size,
                             const size_t hidden_size,
                             const T* past_key,
                             T* present_key,
                             const T* past_value,
                             T* present_value,
                             const bool past_present_share_buffer,
                             const bool packed_qkv,
                             const bool is_prompt,
                             ThreadPool* tp,
                             AllocatorPtr allocator) const {
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads_ / kv_num_heads_);
    const size_t q_input_chunk_length = sequence_length * head_size;
    const size_t kv_input_chunk_length = sequence_length * head_size;
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;

    if (!past_present_share_buffer) {
      if (present_key) {
        memset(static_cast<void*>(present_key), 0,
               batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
      }
      if (present_value) {
        memset(static_cast<void*>(present_value), 0,
               batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
      }
    }

    const size_t kv_loop_len = batch_size * static_cast<size_t>(kv_num_heads_);
    const float alpha = scale_ == 0.0f ? 1.0f / std::sqrt(static_cast<float>(head_size)) : scale_;

    static constexpr size_t kTileSize = 64;

    // Scratch layout per KV-head iteration:
    //   [accum: S×H floats] [row_max: S floats] [row_sum: S floats]
    //   Non-NEON FP16: [K_fp32] [V_fp32] [Q_fp32] [bias_fp32: kTileSize floats]
    //   NEON FP16: no K/V/Q fp32 buffers, [bias_fp32: kTileSize floats] if needed
    const size_t max_total_seqlen = present_buffer_sequence_length;
    size_t per_kv_scratch = 0;

    const size_t accum_off = per_kv_scratch;
    per_kv_scratch += sequence_length * head_size * sizeof(float);
    const size_t row_max_off = per_kv_scratch;
    per_kv_scratch += sequence_length * sizeof(float);
    const size_t row_sum_off = per_kv_scratch;
    per_kv_scratch += sequence_length * sizeof(float);

    // Determine whether we can use the NEON FP16 path
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
    static constexpr bool kHasNeonFp16 = std::is_same_v<T, MLFloat16>;
#else
    static constexpr bool kHasNeonFp16 = false;
#endif

    size_t k_fp32_off = 0, v_fp32_off = 0, q_fp32_off = 0;
    if constexpr (!std::is_same_v<T, float>) {
      if constexpr (!kHasNeonFp16) {
        // Generic FP16 path: need FP32 conversion buffers
        k_fp32_off = per_kv_scratch;
        per_kv_scratch += max_total_seqlen * head_size * sizeof(float);
        v_fp32_off = per_kv_scratch;
        per_kv_scratch += max_total_seqlen * head_size * sizeof(float);
        q_fp32_off = per_kv_scratch;
        per_kv_scratch += sequence_length * head_size * sizeof(float);
      }
    }

    size_t bias_fp32_off = 0;
    if (attention_bias != nullptr && !std::is_same_v<T, float>) {
      bias_fp32_off = per_kv_scratch;
      per_kv_scratch += kTileSize * sizeof(float);
    }

    const size_t total_scratch = per_kv_scratch * kv_loop_len;
    auto scratch_raw = allocator->Alloc(total_scratch);
    BufferUniquePtr scratch_holder(scratch_raw, BufferDeleter(allocator));

    // FP32 output buffer (used by generic FP16 path, not needed for NEON FP16)
    size_t output_fp32_bytes = 0;
    if constexpr (!std::is_same_v<T, float> && !kHasNeonFp16) {
      output_fp32_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * sizeof(float);
    }
    auto output_fp32_raw = allocator->Alloc(output_fp32_bytes);
    BufferUniquePtr output_fp32_holder(output_fp32_raw, BufferDeleter(allocator));
    float* output_fp32 = static_cast<float*>(output_fp32_raw);

    constexpr size_t kSmallInputThreshold = 4;
    ThreadPool* effective_tp = (kv_loop_len <= kSmallInputThreshold) ? nullptr : tp;
    const std::ptrdiff_t num_batches = std::min(
        static_cast<std::ptrdiff_t>(kv_loop_len),
        static_cast<std::ptrdiff_t>(ThreadPool::DegreeOfParallelism(effective_tp)));

    ThreadPool::TryBatchParallelFor(effective_tp, static_cast<std::ptrdiff_t>(kv_loop_len),
                                    [&](std::ptrdiff_t kv_i) {
      const size_t batch_index = static_cast<size_t>(kv_i) / static_cast<size_t>(kv_num_heads_);
      const size_t kv_head_index = static_cast<size_t>(kv_i) % static_cast<size_t>(kv_num_heads_);
      const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
      const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;
      const size_t past_chunk_length = past_seqlen * head_size;

      char* my_scratch = static_cast<char*>(scratch_raw) + static_cast<size_t>(kv_i) * per_kv_scratch;

      // === Concat past K/V ONCE per KV head ===
      const T* k_head;
      if (packed_qkv) {
        k_head = K + packed_batch_stride * batch_index + kv_input_chunk_length * kv_head_index;
      } else {
        k_head = K + kv_input_chunk_length * static_cast<size_t>(kv_i);
      }
      if (nullptr != present_key) {
        k_head = ConcatStateChunkGQA(past_key, k_head, present_key, present_buff_chunk_length,
                                     past_buff_chunk_length, past_chunk_length, kv_input_chunk_length,
                                     past_present_share_buffer, kv_i);
      }

      const T* v_head;
      if (packed_qkv) {
        v_head = V + packed_batch_stride * batch_index + kv_input_chunk_length * kv_head_index;
      } else {
        v_head = V + kv_input_chunk_length * static_cast<size_t>(kv_i);
      }
      if (nullptr != present_value) {
        v_head = ConcatStateChunkGQA(past_value, v_head, present_value, present_buff_chunk_length,
                                     past_buff_chunk_length, past_chunk_length, kv_input_chunk_length,
                                     past_present_share_buffer, kv_i);
      }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)
      if constexpr (kHasNeonFp16) {
        // ========================================================================
        // ARM NEON FP16 path: Q·K dot products in native FP16, FP32 accumulators
        // K/V stay in FP16 — no conversion. 2× dot product throughput with vfmaq_f16.
        // ========================================================================
        const uint16_t* k_fp16 = reinterpret_cast<const uint16_t*>(k_head);
        const uint16_t* v_fp16 = reinterpret_cast<const uint16_t*>(v_head);
        const float alpha_f32 = alpha;

        for (size_t g = 0; g < kv_num_heads_factor; g++) {
          const size_t head_index = kv_head_index * kv_num_heads_factor + g;
          const size_t global_head = batch_index * static_cast<size_t>(num_heads_) + head_index;

          const uint16_t* q_fp16;
          if (packed_qkv) {
            q_fp16 = reinterpret_cast<const uint16_t*>(
                Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index);
          } else {
            q_fp16 = reinterpret_cast<const uint16_t*>(Q + q_input_chunk_length * global_head);
          }

          const T* bias_base = nullptr;
          ptrdiff_t bias_total_seqlen = 0;
          if (attention_bias != nullptr) {
            bias_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
            ptrdiff_t bias_offset = 0;
            const ptrdiff_t attn_matrix_size = static_cast<ptrdiff_t>(sequence_length) * bias_total_seqlen;
            if (attention_bias_shape[0] != 1)
              bias_offset += static_cast<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attn_matrix_size;
            if (attention_bias_shape[1] != 1)
              bias_offset += static_cast<ptrdiff_t>(head_index) * attn_matrix_size;
            bias_base = attention_bias + bias_offset;
          }

          float* accum = reinterpret_cast<float*>(my_scratch + accum_off);
          float* row_max_arr = reinterpret_cast<float*>(my_scratch + row_max_off);
          float* row_sum_arr = reinterpret_cast<float*>(my_scratch + row_sum_off);

          memset(accum, 0, sequence_length * head_size * sizeof(float));
          for (size_t s = 0; s < sequence_length; s++) {
            row_max_arr[s] = -std::numeric_limits<float>::infinity();
            row_sum_arr[s] = 0.0f;
          }

          const size_t max_causal = past_seqlen + sequence_length;

          for (size_t t_start = 0; t_start < max_causal; t_start += kTileSize) {
            const size_t t_end_tile = std::min(t_start + kTileSize, max_causal);

            for (size_t seq = 0; seq < sequence_length; seq++) {
              const size_t seq_causal_length = past_seqlen + seq + 1;
              const bool apply_local = local_window_size_ >= 0 &&
                                       seq_causal_length > static_cast<size_t>(local_window_size_);
              const size_t local_start = apply_local
                                             ? seq_causal_length - static_cast<size_t>(local_window_size_)
                                             : 0;
              const size_t eff_start = std::max(t_start, local_start);
              const size_t eff_end = std::min(t_end_tile, seq_causal_length);
              if (eff_start >= eff_end) continue;

              const size_t tile_len = eff_end - eff_start;
              const uint16_t* q_row = q_fp16 + seq * head_size;
              float* accum_row = accum + seq * head_size;

              // --- NEON FP16 dot product: Q·K' ---
              // Compute tile_scores[j] = dot(q_row, K[eff_start+j]) * alpha
              float tile_scores[kTileSize];
              for (size_t j = 0; j < tile_len; j++) {
                const uint16_t* k_row = k_fp16 + (eff_start + j) * head_size;

                // Process head_size elements in chunks of 8 using float16x8_t
                float16x8_t acc_vec = vreinterpretq_f16_f32(vdupq_n_f32(0.0f));
                size_t d = 0;
                for (; d + 8 <= head_size; d += 8) {
                  float16x8_t q_vec = vreinterpretq_f16_u16(vld1q_u16(q_row + d));
                  float16x8_t k_vec = vreinterpretq_f16_u16(vld1q_u16(k_row + d));
                  acc_vec = vfmaq_f16(acc_vec, q_vec, k_vec);
                }

                // Horizontal reduce 8 → 1 (pairwise add in FP16, then extract)
                float16x4_t sum4 = vadd_f16(vget_low_f16(acc_vec), vget_high_f16(acc_vec));
                sum4 = vpadd_f16(sum4, sum4);
                sum4 = vpadd_f16(sum4, sum4);
                float dot = vgetq_lane_f32(vcvt_f32_f16(sum4), 0);

                // Handle tail elements (head_size not multiple of 8)
                for (; d < head_size; d++) {
                  float q_val = MLFloat16::FromBits(q_row[d]).ToFloat();
                  float k_val = MLFloat16::FromBits(k_row[d]).ToFloat();
                  dot += q_val * k_val;
                }

                tile_scores[j] = dot * alpha_f32;
              }

              // Softcap
              if (softcap_ > 0.f) {
                const float inv_cap = 1.0f / softcap_;
                for (size_t j = 0; j < tile_len; j++) {
                  tile_scores[j] = softcap_ * std::tanh(tile_scores[j] * inv_cap);
                }
              }

              // Attention bias
              if (bias_base != nullptr) {
                const uint16_t* bias_row = reinterpret_cast<const uint16_t*>(bias_base + seq * bias_total_seqlen);
                float* bias_fp32 = reinterpret_cast<float*>(my_scratch + bias_fp32_off);
                MlasConvertHalfToFloatBuffer(
                    reinterpret_cast<const MLFloat16*>(bias_row + eff_start), bias_fp32, tile_len);
                for (size_t j = 0; j < tile_len; j++) {
                  tile_scores[j] += bias_fp32[j];
                }
              }

              // --- Online softmax update (FP32 for numerical stability) ---
              float tile_max = tile_scores[0];
              for (size_t j = 1; j < tile_len; j++) {
                tile_max = std::max(tile_max, tile_scores[j]);
              }

              const float new_max = std::max(row_max_arr[seq], tile_max);
              const float correction =
                  (row_max_arr[seq] > -std::numeric_limits<float>::infinity())
                      ? std::exp(row_max_arr[seq] - new_max)
                      : 0.0f;

              // Rescale FP32 accumulator by correction factor using NEON float32x4_t
              {
                float32x4_t corr_vec = vdupq_n_f32(correction);
                size_t d = 0;
                for (; d + 4 <= head_size; d += 4) {
                  float32x4_t a = vld1q_f32(accum_row + d);
                  a = vmulq_f32(a, corr_vec);
                  vst1q_f32(accum_row + d, a);
                }
                for (; d < head_size; d++) {
                  accum_row[d] *= correction;
                }
              }
              row_sum_arr[seq] *= correction;

              // --- Accumulate weighted V in FP32, loading V as FP16 and widening ---
              float tile_sum = 0.0f;
              for (size_t j = 0; j < tile_len; j++) {
                const float w = std::exp(tile_scores[j] - new_max);
                tile_sum += w;
                const uint16_t* v_row = v_fp16 + (eff_start + j) * head_size;

                // FMA: accum_row[d] += w * V[d], widening FP16 V to FP32 in groups of 4
                float32x4_t w_vec = vdupq_n_f32(w);
                size_t d = 0;
                for (; d + 8 <= head_size; d += 8) {
                  // Load 8 FP16 values, widen to two float32x4_t, FMA into accumulator
                  float16x8_t v_f16 = vreinterpretq_f16_u16(vld1q_u16(v_row + d));
                  float32x4_t v_lo = vcvt_f32_f16(vget_low_f16(v_f16));
                  float32x4_t v_hi = vcvt_f32_f16(vget_high_f16(v_f16));
                  float32x4_t a_lo = vld1q_f32(accum_row + d);
                  float32x4_t a_hi = vld1q_f32(accum_row + d + 4);
                  a_lo = vfmaq_f32(a_lo, w_vec, v_lo);
                  a_hi = vfmaq_f32(a_hi, w_vec, v_hi);
                  vst1q_f32(accum_row + d, a_lo);
                  vst1q_f32(accum_row + d + 4, a_hi);
                }
                for (; d + 4 <= head_size; d += 4) {
                  float16x4_t v_f16 = vreinterpret_f16_u16(vld1_u16(v_row + d));
                  float32x4_t v_f32 = vcvt_f32_f16(v_f16);
                  float32x4_t a = vld1q_f32(accum_row + d);
                  a = vfmaq_f32(a, w_vec, v_f32);
                  vst1q_f32(accum_row + d, a);
                }
                for (; d < head_size; d++) {
                  accum_row[d] += w * MLFloat16::FromBits(v_row[d]).ToFloat();
                }
              }

              row_sum_arr[seq] += tile_sum;
              row_max_arr[seq] = new_max;
            }
          }

          // === Finalization ===
          for (size_t seq = 0; seq < sequence_length; seq++) {
            float* accum_row = accum + seq * head_size;

            if (use_smooth_softmax_ || head_sink != nullptr) {
              const float sink_val = (head_sink != nullptr)
                                         ? static_cast<float>(head_sink[head_index])
                                         : 0.0f;
              const float sink_new_max = std::max(row_max_arr[seq], sink_val);
              const float corr =
                  (row_max_arr[seq] > -std::numeric_limits<float>::infinity())
                      ? std::exp(row_max_arr[seq] - sink_new_max)
                      : 0.0f;
              float32x4_t corr_vec = vdupq_n_f32(corr);
              size_t d = 0;
              for (; d + 4 <= head_size; d += 4) {
                float32x4_t a = vld1q_f32(accum_row + d);
                vst1q_f32(accum_row + d, vmulq_f32(a, corr_vec));
              }
              for (; d < head_size; d++) accum_row[d] *= corr;
              row_sum_arr[seq] = corr * row_sum_arr[seq] + std::exp(sink_val - sink_new_max);
            }

            // Normalize and convert FP32 → FP16 directly into output
            T* out = output +
                     (batch_index * sequence_length * static_cast<size_t>(num_heads_) + head_index) * head_size;
            T* out_row = out + seq * hidden_size;

            if (row_sum_arr[seq] > 0.0f) {
              const float inv_sum = 1.0f / row_sum_arr[seq];
              float32x4_t inv_vec = vdupq_n_f32(inv_sum);
              size_t d = 0;
              for (; d + 8 <= head_size; d += 8) {
                float32x4_t a_lo = vmulq_f32(vld1q_f32(accum_row + d), inv_vec);
                float32x4_t a_hi = vmulq_f32(vld1q_f32(accum_row + d + 4), inv_vec);
                // Convert FP32 → FP16 and store
                float16x4_t h_lo = vcvt_f16_f32(a_lo);
                float16x4_t h_hi = vcvt_f16_f32(a_hi);
                float16x8_t h8 = vcombine_f16(h_lo, h_hi);
                vst1q_u16(reinterpret_cast<uint16_t*>(out_row) + d, vreinterpretq_u16_f16(h8));
              }
              for (; d + 4 <= head_size; d += 4) {
                float32x4_t a = vmulq_f32(vld1q_f32(accum_row + d), inv_vec);
                float16x4_t h = vcvt_f16_f32(a);
                vst1_u16(reinterpret_cast<uint16_t*>(out_row) + d, vreinterpret_u16_f16(h));
              }
              for (; d < head_size; d++) {
                reinterpret_cast<uint16_t*>(out_row)[d] = MLFloat16(accum_row[d] * inv_sum).val;
              }
            } else {
              memset(out_row, 0, head_size * sizeof(T));
            }
          }
        }  // end GQA group loop
      } else  // NOLINT(readability/braces)
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
      {
        // ========================================================================
        // Generic path: FP32 scalar (T=float) or FP16→FP32 conversion (non-ARM)
        // ========================================================================
        const float* k_data;
        const float* v_data;
        if constexpr (std::is_same_v<T, float>) {
          k_data = k_head;
          v_data = v_head;
        } else {
          float* k_buf = reinterpret_cast<float*>(my_scratch + k_fp32_off);
          MlasConvertHalfToFloatBuffer(k_head, k_buf, head_size * total_seqlen);
          k_data = k_buf;
          float* v_buf = reinterpret_cast<float*>(my_scratch + v_fp32_off);
          MlasConvertHalfToFloatBuffer(v_head, v_buf, head_size * total_seqlen);
          v_data = v_buf;
        }

        for (size_t g = 0; g < kv_num_heads_factor; g++) {
          const size_t head_index = kv_head_index * kv_num_heads_factor + g;
          const size_t global_head = batch_index * static_cast<size_t>(num_heads_) + head_index;

          const T* q_head_t;
          if (packed_qkv) {
            q_head_t = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
          } else {
            q_head_t = Q + q_input_chunk_length * global_head;
          }

          const float* q_fp32;
          if constexpr (std::is_same_v<T, float>) {
            q_fp32 = q_head_t;
          } else {
            float* q_buf = reinterpret_cast<float*>(my_scratch + q_fp32_off);
            MlasConvertHalfToFloatBuffer(q_head_t, q_buf, head_size * sequence_length);
            q_fp32 = q_buf;
          }

          const T* bias_base = nullptr;
          ptrdiff_t bias_total_seqlen = 0;
          if (attention_bias != nullptr) {
            bias_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
            ptrdiff_t bias_offset = 0;
            const ptrdiff_t attn_matrix_size = static_cast<ptrdiff_t>(sequence_length) * bias_total_seqlen;
            if (attention_bias_shape[0] != 1)
              bias_offset += static_cast<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attn_matrix_size;
            if (attention_bias_shape[1] != 1)
              bias_offset += static_cast<ptrdiff_t>(head_index) * attn_matrix_size;
            bias_base = attention_bias + bias_offset;
          }

          float* accum = reinterpret_cast<float*>(my_scratch + accum_off);
          float* row_max_arr = reinterpret_cast<float*>(my_scratch + row_max_off);
          float* row_sum_arr = reinterpret_cast<float*>(my_scratch + row_sum_off);

          memset(accum, 0, sequence_length * head_size * sizeof(float));
          for (size_t s = 0; s < sequence_length; s++) {
            row_max_arr[s] = -std::numeric_limits<float>::infinity();
            row_sum_arr[s] = 0.0f;
          }

          const size_t max_causal = past_seqlen + sequence_length;

          for (size_t t_start = 0; t_start < max_causal; t_start += kTileSize) {
            const size_t t_end_tile = std::min(t_start + kTileSize, max_causal);

            for (size_t seq = 0; seq < sequence_length; seq++) {
              const size_t seq_causal_length = past_seqlen + seq + 1;
              const bool apply_local = local_window_size_ >= 0 &&
                                       seq_causal_length > static_cast<size_t>(local_window_size_);
              const size_t local_start = apply_local
                                             ? seq_causal_length - static_cast<size_t>(local_window_size_)
                                             : 0;
              const size_t eff_start = std::max(t_start, local_start);
              const size_t eff_end = std::min(t_end_tile, seq_causal_length);
              if (eff_start >= eff_end) continue;

              const size_t tile_len = eff_end - eff_start;
              const float* q_row = q_fp32 + seq * head_size;
              float* accum_row = accum + seq * head_size;

              float tile_scores[kTileSize];
              for (size_t j = 0; j < tile_len; j++) {
                const float* k_row = k_data + (eff_start + j) * head_size;
                float dot = 0.0f;
                for (size_t d = 0; d < head_size; d++) {
                  dot += q_row[d] * k_row[d];
                }
                tile_scores[j] = dot * alpha;
              }

              if (softcap_ > 0.f) {
                const float inv_cap = 1.0f / softcap_;
                for (size_t j = 0; j < tile_len; j++) {
                  tile_scores[j] = softcap_ * std::tanh(tile_scores[j] * inv_cap);
                }
              }

              if (bias_base != nullptr) {
                const T* bias_row = bias_base + seq * bias_total_seqlen;
                if constexpr (std::is_same_v<T, float>) {
                  for (size_t j = 0; j < tile_len; j++) {
                    tile_scores[j] += bias_row[eff_start + j];
                  }
                } else {
                  float* bias_fp32 = reinterpret_cast<float*>(my_scratch + bias_fp32_off);
                  MlasConvertHalfToFloatBuffer(bias_row + eff_start, bias_fp32, tile_len);
                  for (size_t j = 0; j < tile_len; j++) {
                    tile_scores[j] += bias_fp32[j];
                  }
                }
              }

              float tile_max = tile_scores[0];
              for (size_t j = 1; j < tile_len; j++) {
                tile_max = std::max(tile_max, tile_scores[j]);
              }

              const float new_max = std::max(row_max_arr[seq], tile_max);
              const float correction =
                  (row_max_arr[seq] > -std::numeric_limits<float>::infinity())
                      ? std::exp(row_max_arr[seq] - new_max)
                      : 0.0f;

              for (size_t d = 0; d < head_size; d++) {
                accum_row[d] *= correction;
              }
              row_sum_arr[seq] *= correction;

              float tile_sum = 0.0f;
              for (size_t j = 0; j < tile_len; j++) {
                const float w = std::exp(tile_scores[j] - new_max);
                tile_sum += w;
                const float* v_row = v_data + (eff_start + j) * head_size;
                for (size_t d = 0; d < head_size; d++) {
                  accum_row[d] += w * v_row[d];
                }
              }

              row_sum_arr[seq] += tile_sum;
              row_max_arr[seq] = new_max;
            }
          }

          for (size_t seq = 0; seq < sequence_length; seq++) {
            float* accum_row = accum + seq * head_size;

            if (use_smooth_softmax_ || head_sink != nullptr) {
              const float sink_val = (head_sink != nullptr)
                                         ? static_cast<float>(head_sink[head_index])
                                         : 0.0f;
              const float sink_new_max = std::max(row_max_arr[seq], sink_val);
              const float corr =
                  (row_max_arr[seq] > -std::numeric_limits<float>::infinity())
                      ? std::exp(row_max_arr[seq] - sink_new_max)
                      : 0.0f;
              for (size_t d = 0; d < head_size; d++) accum_row[d] *= corr;
              row_sum_arr[seq] = corr * row_sum_arr[seq] + std::exp(sink_val - sink_new_max);
            }

            if (row_sum_arr[seq] > 0.0f) {
              const float inv_sum = 1.0f / row_sum_arr[seq];
              for (size_t d = 0; d < head_size; d++) {
                accum_row[d] *= inv_sum;
              }
            }
          }

          if constexpr (std::is_same_v<T, float>) {
            float* out = output +
                         (batch_index * sequence_length * static_cast<size_t>(num_heads_) + head_index) * head_size;
            for (size_t s = 0; s < sequence_length; s++) {
              memcpy(out + s * hidden_size, accum + s * head_size, head_size * sizeof(float));
            }
          } else {
            float* out_fp32 = output_fp32 +
                              (batch_index * sequence_length * static_cast<size_t>(num_heads_) + head_index) * head_size;
            for (size_t s = 0; s < sequence_length; s++) {
              memcpy(out_fp32 + s * hidden_size, accum + s * head_size, head_size * sizeof(float));
            }
          }
        }
      }
    }, num_batches);

    // Convert FP32 output back to FP16 (generic path only)
    if constexpr (!std::is_same_v<T, float> && !kHasNeonFp16) {
      MlasConvertFloatToHalfBuffer(output_fp32, output,
                                   SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size);
    }
  }

  // Fused computation of attention probs and V x attention score in a single parallel region.
  // This eliminates one fork-join barrier compared to calling ComputeAttentionProbs and
  // ComputeVxAttentionScore separately. For each (batch, head), we:
  //   1. Concat past K, compute Q*K' + bias + softmax → attention_probs
  //   2. Concat past V, compute attention_probs * V → output
  // Since each head's attention_probs are independent, both phases can be done in one iteration.
  template <typename T, typename U>
  void ComputeFusedAttention(U* attention_probs,
                             const T* Q,
                             const T* K,
                             const T* V,
                             const T* head_sink,
                             const int32_t* seqlens_k,
                             const T* attention_bias,
                             T* output,
                             T* output_qk,
                             const size_t batch_size,
                             const size_t sequence_length,
                             const size_t total_sequence_length,
                             const gsl::span<const int64_t> attention_bias_shape,
                             const size_t past_buffer_sequence_length,
                             const size_t present_buffer_sequence_length,
                             const size_t head_size,
                             const size_t hidden_size,
                             const T* past_key,
                             T* present_key,
                             const T* past_value,
                             T* present_value,
                             const bool past_present_share_buffer,
                             const bool packed_qkv,
                             const bool is_prompt,
                             ThreadPool* tp,
                             AllocatorPtr allocator) const {
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = sequence_length * head_size;
    const size_t kv_input_chunk_length = sequence_length * head_size;
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;

    // Zero present KV buffers if not sharing with past
    if (!past_present_share_buffer) {
      if (present_key) {
        memset(static_cast<void*>(present_key), 0,
               batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
      }
      if (present_value) {
        memset(static_cast<void*>(present_value), 0,
               batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
      }
    }

    const size_t loop_len = batch_size * num_heads_;
    const float alpha = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    // Pre-allocate fp32 output buffer for fp16 input with fp32 accumulation
    size_t output_fp32_bytes = 0;
    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      output_fp32_bytes = SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size * sizeof(float);
    }
    auto output_fp32 = allocator->Alloc(output_fp32_bytes);
    BufferUniquePtr output_fp32_buffer(output_fp32, BufferDeleter(allocator));

    // Pre-allocate per-thread scratch buffers for fp16→fp32 conversion to avoid
    // allocator contention inside the parallel loop.
    // We allocate per-iteration (per head) scratch indexed by loop iteration i.
    const size_t max_total_seqlen = present_buffer_sequence_length;
    const size_t per_iter_qk_fp32_bytes =
        (std::is_same<T, MLFloat16>::value && !std::is_same<U, MLFloat16>::value)
            ? (head_size * (sequence_length + max_total_seqlen) * sizeof(float) +
               head_size * max_total_seqlen * sizeof(float))
            : 0;
    const size_t per_iter_bias_fp32_bytes =
        (attention_bias != nullptr && !std::is_same_v<U, T>)
            ? (max_total_seqlen * sizeof(float))
            : 0;
    const size_t per_iter_scratch_bytes = per_iter_qk_fp32_bytes + per_iter_bias_fp32_bytes;
    const size_t total_scratch_bytes = per_iter_scratch_bytes * loop_len;
    void* iter_scratch_raw = nullptr;
    BufferUniquePtr iter_scratch_holder;
    if (total_scratch_bytes > 0) {
      iter_scratch_raw = allocator->Alloc(total_scratch_bytes);
      iter_scratch_holder = BufferUniquePtr(iter_scratch_raw, BufferDeleter(allocator));
    }

    // For small inputs, bypass thread pool entirely to avoid scheduling overhead.
    // The threshold is chosen so that the fork-join barrier cost (~5-10µs) would
    // dominate the actual compute time.
    constexpr size_t kSmallInputThreshold = 4;  // batch_size * num_heads
    ThreadPool* effective_tp = (loop_len <= kSmallInputThreshold) ? nullptr : tp;

    // Use TryBatchParallelFor instead of TryParallelFor to bypass the Eigen cost model.
    // The cost model's startup overhead threshold (100K cycles) is too conservative for
    // this loop where each iteration is a full GEMM + softmax + GEMM — always worth
    // parallelizing when we have enough iterations. TryBatchParallelFor directly partitions
    // work across all available threads without cost model evaluation.
    const std::ptrdiff_t num_batches = std::min(
        static_cast<std::ptrdiff_t>(loop_len),
        static_cast<std::ptrdiff_t>(ThreadPool::DegreeOfParallelism(effective_tp)));

    ThreadPool::TryBatchParallelFor(effective_tp, loop_len, [&](std::ptrdiff_t i) {
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;
        const size_t past_chunk_length = past_seqlen * head_size;

        // Get per-iteration scratch buffer (indexed by i, no thread-id needed)
        char* my_scratch = (iter_scratch_raw != nullptr)
                               ? static_cast<char*>(iter_scratch_raw) + i * per_iter_scratch_bytes
                               : nullptr;

        // ============ Phase 1: Q*K' + softmax → attention_probs ============

        const ptrdiff_t probs_output_offset = SafeInt<ptrdiff_t>(i) * sequence_length * present_buffer_sequence_length;
        U* probs_output = attention_probs + probs_output_offset;
        T* output_qk_thread = nullptr;
        if (output_qk != nullptr) {
          const ptrdiff_t output_qk_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * (batch_index * num_heads_ + head_index);
          output_qk_thread = output_qk + output_qk_offset;
        }

        // Attention bias offset
        const T* attention_bias_thread = nullptr;
        ptrdiff_t attention_total_seqlen = 0;
        if (attention_bias != nullptr) {
          ptrdiff_t attention_bias_offset = 0;
          attention_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
          const ptrdiff_t attention_matrix_size = sequence_length * attention_total_seqlen;
          if (attention_bias_shape[0] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attention_matrix_size;
          }
          if (attention_bias_shape[1] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(head_index) * attention_matrix_size;
          }
          attention_bias_thread = attention_bias + attention_bias_offset;
        }

        // Concat past K and get present K pointer
        const T* k_head;
        if (packed_qkv) {
          k_head = K + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          k_head = K + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_key) {
          k_head = ConcatStateChunkGQA(past_key, k_head, present_key, present_buff_chunk_length, past_buff_chunk_length,
                                       past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                       i / kv_num_heads_factor);
        }

        // Q pointer
        const T* q_head;
        if (packed_qkv) {
          q_head = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q_head = Q + q_input_chunk_length * i;
        }

        // Q*K' GEMM
        if constexpr (std::is_same<T, float>::value) {
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q_head,
                                          static_cast<int>(head_size), k_head, static_cast<int>(head_size), 0.0f,
                                          probs_output, static_cast<int>(present_buffer_sequence_length), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          MlasGemm(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size,
                   q_head, static_cast<int>(head_size), k_head, static_cast<int>(head_size), probs_output,
                   static_cast<int>(present_buffer_sequence_length),
                   MLFloat16(alpha).val, static_cast<uint16_t>(0), nullptr);
        } else {
          // fp16 input, fp32 accumulation: use pre-allocated scratch
          float* q_fp32 = reinterpret_cast<float*>(my_scratch);
          MlasConvertHalfToFloatBuffer(q_head, q_fp32, head_size * sequence_length);
          float* k_fp32 = q_fp32 + head_size * sequence_length;
          MlasConvertHalfToFloatBuffer(k_head, k_fp32, head_size * total_seqlen);

          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q_fp32,
                                          static_cast<int>(head_size), k_fp32, static_cast<int>(head_size), 0.0f,
                                          probs_output, static_cast<int>(present_buffer_sequence_length), nullptr);
        }

        // Attention bias fp32 buffer (for fp16 bias with fp32 probs)
        float* attention_bias_thread_fp32 = nullptr;
        if (attention_bias_thread != nullptr && !std::is_same_v<U, T>) {
          if constexpr (!std::is_same_v<U, T>) {
            static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);
            attention_bias_thread_fp32 = reinterpret_cast<float*>(
                my_scratch + per_iter_qk_fp32_bytes);
          }
        }

        // Softmax per sequence position
        U* output_softmax = probs_output;
        const T* bias_ptr = attention_bias_thread;
        T* qk_ptr = output_qk_thread;
        for (size_t seq = 0; seq < sequence_length; seq++) {
          size_t seq_causal_length = past_seqlen + seq + 1;
          const bool should_apply_local_window = local_window_size_ >= 0 &&
                                                 seq_causal_length > static_cast<size_t>(local_window_size_);
          const size_t start_offset = should_apply_local_window ? seq_causal_length - local_window_size_ : 0;
          const size_t window_size = should_apply_local_window ? local_window_size_ : seq_causal_length;

          if (should_apply_local_window) {
            for (size_t total_seq_id = 0; total_seq_id < seq_causal_length - local_window_size_; total_seq_id++) {
              if constexpr (std::is_same<U, float>::value) {
                output_softmax[total_seq_id] = 0.f;
              } else {
                output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
              }
            }
          }

          if (softcap_ > 0.f) {
            ComputeAttentionSoftcapInplace(output_softmax + start_offset, static_cast<int>(window_size),
                                           static_cast<U>(softcap_));
          }

          if (bias_ptr != nullptr) {
            if constexpr (std::is_same_v<U, T>) {
              ApplyAttentionBias(output_softmax + start_offset, bias_ptr + start_offset,
                                 static_cast<int>(window_size));
            } else {
              static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);
              MlasConvertHalfToFloatBuffer(bias_ptr + start_offset, attention_bias_thread_fp32, window_size);
              ApplyAttentionBias(output_softmax + start_offset, attention_bias_thread_fp32, static_cast<int>(window_size));
            }
          }

          for (size_t total_seq_id = seq_causal_length; total_seq_id < total_seqlen; total_seq_id++) {
            if constexpr (std::is_same<U, float>::value) {
              output_softmax[total_seq_id] = 0.f;
            } else {
              output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
            }
          }

          if (qk_output_ == static_cast<int>(QKOutputType::BEFORE_SOFTMAX)) {
            WriteOutputQKHeadChunk(qk_ptr, output_softmax, total_sequence_length);
          }

          if (use_smooth_softmax_ || head_sink != nullptr) {
            float sink = (head_sink != nullptr) ? static_cast<float>(head_sink[head_index]) : 0.0f;
            ComputeSmoothSoftmaxInplace(output_softmax + start_offset, static_cast<int>(window_size), sink, nullptr);
          } else {
            ComputeAttentionSoftmaxInplace(output_softmax + start_offset, 1, static_cast<int>(window_size), nullptr);
          }

          if (qk_output_ == static_cast<int>(QKOutputType::AFTER_SOFTMAX)) {
            WriteOutputQKHeadChunk(qk_ptr, output_softmax, total_sequence_length);
          }

          output_softmax += present_buffer_sequence_length;
          if (bias_ptr != nullptr) {
            bias_ptr += attention_total_seqlen;
          }
          if (qk_ptr != nullptr) {
            qk_ptr += total_sequence_length;
          }
        }

        // ============ Phase 2: attention_probs * V → output ============

        // Concat past V and get present V pointer
        const T* v_head;
        if (packed_qkv) {
          v_head = V + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          v_head = V + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_value) {
          v_head = ConcatStateChunkGQA(past_value, v_head, present_value, present_buff_chunk_length, past_buff_chunk_length,
                                       past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                       i / kv_num_heads_factor);
        }

        ptrdiff_t attn_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * i;

        if constexpr (std::is_same<T, float>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, attention_probs + attn_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v_head,
                                          static_cast<int>(head_size), 0.0f, output_current,
                                          static_cast<int>(hidden_size), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          MlasGemm(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                   attention_probs + attn_probs_offset, static_cast<int>(present_buffer_sequence_length),
                   v_head, static_cast<int>(head_size), output_current, static_cast<int>(hidden_size),
                   MLFloat16(1.0f).val, static_cast<uint16_t>(0), nullptr);
        } else {
          // fp16 input, fp32 accumulation: reuse pre-allocated scratch for V fp32
          float* v_fp32_ptr = reinterpret_cast<float*>(my_scratch) +
                              head_size * (sequence_length + max_total_seqlen);
          MlasConvertHalfToFloatBuffer(v_head, v_fp32_ptr, head_size * total_seqlen);

          float* output_fp32_current = static_cast<float*>(output_fp32) +
                                       (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, attention_probs + attn_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v_fp32_ptr,
                                          static_cast<int>(head_size), 0.0f, output_fp32_current,
                                          static_cast<int>(hidden_size), nullptr);
        }
    }, num_batches);

    // Convert fp32 output back to fp16 if needed
    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      MlasConvertFloatToHalfBuffer(static_cast<float*>(output_fp32),
                                   output,
                                   SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size);
    }
  }

  // Original unfused methods kept for reference but no longer called from ApplyAttention.
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  // If T is float32, U is float32. If T is float16, U could be float16 or float32.
  template <typename T, typename U>
  void ComputeAttentionProbs(U* attention_probs,                                   // output buffer with size BxNxSxT
                             const T* Q,                                           // Q data. Its size is BxNxSxH
                             const T* K,                                           // k data. Its size is BxNxLxH
                             const T* head_sink,                                   // for smooth softmax. Its size is N.
                             const int32_t* seqlens_k,                             // total - 1 sequence lengths tensor
                             const T* attention_bias,                              // optional attention bias
                             const size_t batch_size,                              // batch size of self-attention
                             const size_t sequence_length,                         // sequence length of self-attention (S)
                             const size_t total_sequence_length,                   // total sequence length (T)
                             const gsl::span<const int64_t> attention_bias_shape,  // shape of the attention bias
                             const size_t past_buffer_sequence_length,             // sequence length of past state
                             const size_t present_buffer_sequence_length,          // sequence length of present state
                             const size_t head_size,                               // head size of self-attention
                             const T* past_key,                                    // past key only
                             T* present_key,                                       // present key only
                             T* output_qk,                                         // output QK buffer
                             const bool past_present_share_buffer,                 // whether present key and value share the same buffer
                             const bool packed_qkv,                                // whether Q, K, V are packed
                             const bool is_prompt,                                 // whether it is prompt
                             ThreadPool* tp,                                       // thread pool
                             AllocatorPtr allocator) const {                       // allocator for temporary buffer
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = sequence_length * head_size;                      // S x H
    const size_t kv_input_chunk_length = sequence_length * head_size;                     // L x H
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;        // L x H
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset((void*)present_key,
             0,
             batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    const size_t loop_len = batch_size * num_heads_;
    const float alpha = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    TensorOpCost unit_cost;
    const ptrdiff_t probs_matrix_bytes =
        SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * sizeof(T);
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
    unit_cost.bytes_loaded =
        static_cast<double>((sequence_length + present_buffer_sequence_length) * head_size * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

    unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes);
    unit_cost.bytes_stored += static_cast<double>(probs_matrix_bytes);

    if (present_key) {
      double bytes_to_copy_key = static_cast<double>(sizeof(T) * present_buff_chunk_length);
      unit_cost.bytes_loaded += bytes_to_copy_key;
      unit_cost.bytes_stored += bytes_to_copy_key;
    }

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;  // Assume no padding sequence length
        const size_t past_chunk_length = past_seqlen * head_size;

        const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * sequence_length * present_buffer_sequence_length;
        U* output = attention_probs + output_offset;
        T* output_qk_thread = nullptr;
        if (output_qk != nullptr) {
          const ptrdiff_t output_qk_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * (batch_index * num_heads_ + head_index);
          output_qk_thread = output_qk + output_qk_offset;
        }

        // Compute attention bias offset based on the batch and head indexes
        // Attention bias is of shape (B or 1, H or 1, S, T) so handle broadcasting
        const T* attention_bias_thread = nullptr;
        ptrdiff_t attention_total_seqlen = 0;
        if (attention_bias != nullptr) {
          ptrdiff_t attention_bias_offset = 0;
          attention_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
          const ptrdiff_t attention_matrix_size = sequence_length * attention_total_seqlen;
          if (attention_bias_shape[0] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] * attention_matrix_size;
          }
          if (attention_bias_shape[1] != 1) {
            attention_bias_offset += SafeInt<ptrdiff_t>(head_index) * attention_matrix_size;
          }

          attention_bias_thread = attention_bias + attention_bias_offset;
        }

        const T* k;
        if (packed_qkv) {
          k = K + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          k = K + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_key) {
          k = ConcatStateChunkGQA(past_key, k, present_key, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                  i / kv_num_heads_factor);
        }

        // Compute Q*K' + AttentionMask
        //                     original                 transposed             each iteration
        // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
        // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
        const T* q;
        if (packed_qkv) {
          q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        } else {
          q = Q + q_input_chunk_length * i;
        }

        if constexpr (std::is_same<T, float>::value) {
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q,
                                          static_cast<int>(head_size), k, static_cast<int>(head_size), 0.0f /*bata*/,
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          MlasGemm(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size,
                   q, static_cast<int>(head_size), k, static_cast<int>(head_size), output,
                   static_cast<int>(present_buffer_sequence_length),
                   MLFloat16(alpha).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = head_size * (sequence_length + total_seqlen) * sizeof(float);
          auto q_k_fp32 = allocator->Alloc(bytes);
          BufferUniquePtr scratch_buffer(q_k_fp32, BufferDeleter(allocator));

          float* q_fp32 = static_cast<float*>(q_k_fp32);
          MlasConvertHalfToFloatBuffer(q, q_fp32, head_size * sequence_length);

          float* k_fp32 = q_fp32 + head_size * sequence_length;
          MlasConvertHalfToFloatBuffer(k, k_fp32, head_size * total_seqlen);

          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q_fp32,
                                          static_cast<int>(head_size), k_fp32, static_cast<int>(head_size), 0.0f /*bata*/,
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr);
        }

        // Pre-allocate buffer for attention mask to avoid allocating it for every processed token
        float* attention_bias_thread_fp32 = nullptr;
        if (attention_bias_thread != nullptr) {
          if constexpr (!std::is_same_v<U, T>) {
            static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);

            size_t bytes = attention_total_seqlen * sizeof(float);
            attention_bias_thread_fp32 = static_cast<float*>(allocator->Alloc(bytes));
          }
        }
        BufferUniquePtr scratch_buffer(attention_bias_thread_fp32, BufferDeleter(allocator));

        // compute Softmax
        U* output_softmax = output;
        for (size_t seq = 0; seq < sequence_length; seq++) {
          size_t seq_causal_length = past_seqlen + seq + 1;

          const bool should_apply_local_window = local_window_size_ >= 0 &&
                                                 seq_causal_length > static_cast<size_t>(local_window_size_);

          const size_t start_offset = should_apply_local_window ? seq_causal_length - local_window_size_ : 0;
          const size_t window_size = should_apply_local_window ? local_window_size_ : seq_causal_length;

          // Mask everything before local window, if local window should be applied
          if (should_apply_local_window) {
            for (size_t total_seq_id = 0; total_seq_id < seq_causal_length - local_window_size_; total_seq_id++) {
              if constexpr (std::is_same<U, float>::value) {
                output_softmax[total_seq_id] = 0.f;
              } else {
                output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
              }
            }
          }

          if (softcap_ > 0.f) {
            ComputeAttentionSoftcapInplace(output_softmax + start_offset, static_cast<int>(window_size),
                                           static_cast<U>(softcap_));
          }

          // Add attention bias to QxK' if provided
          // TODO (#23982): Implement bias addition during softmax computation in GQA CPU operator
          if (attention_bias_thread != nullptr) {
            if constexpr (std::is_same_v<U, T>) {
              ApplyAttentionBias(output_softmax + start_offset, attention_bias_thread + start_offset,
                                 static_cast<int>(window_size));
            } else {
              static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);

              MlasConvertHalfToFloatBuffer(attention_bias_thread + start_offset, attention_bias_thread_fp32, window_size);
              ApplyAttentionBias(output_softmax + start_offset, attention_bias_thread_fp32, static_cast<int>(window_size));
            }
          }

          // set causal [seq_causal_length, total_seqlen) to 0.f
          for (size_t total_seq_id = seq_causal_length; total_seq_id < total_seqlen; total_seq_id++) {
            if constexpr (std::is_same<U, float>::value) {
              output_softmax[total_seq_id] = 0.f;
            } else {
              output_softmax[total_seq_id] = MLFloat16::FromBits(static_cast<uint16_t>(0));
            }
          }

          if (qk_output_ == static_cast<int>(QKOutputType::BEFORE_SOFTMAX)) {
            WriteOutputQKHeadChunk(output_qk_thread, output_softmax, total_sequence_length);
          }

          if (use_smooth_softmax_ || head_sink != nullptr) {
            float sink = (head_sink != nullptr) ? static_cast<float>(head_sink[head_index]) : 0.0f;
            ComputeSmoothSoftmaxInplace(output_softmax + start_offset, static_cast<int>(window_size), sink, nullptr);
          } else {
            ComputeAttentionSoftmaxInplace(output_softmax + start_offset, 1, static_cast<int>(window_size), nullptr);
          }

          if (qk_output_ == static_cast<int>(QKOutputType::AFTER_SOFTMAX)) {
            WriteOutputQKHeadChunk(output_qk_thread, output_softmax, total_sequence_length);
          }

          output_softmax += present_buffer_sequence_length;

          if (attention_bias_thread != nullptr) {
            attention_bias_thread += attention_total_seqlen;
          }

          if (output_qk_thread != nullptr) {
            output_qk_thread += total_sequence_length;
          }
        }
      }
    });
  }

  template <typename T, typename U>
  void ComputeVxAttentionScore(T* output,                                    // buffer for the result with size BxSxNxH
                               const U* attention_probs,                     // Attention probs with size BxNxSxT
                               const T* V,                                   // V value with size BxN_kvxSxH
                               const int32_t* seqlens_k,                     // total - 1 sequence lengths tensor
                               const size_t batch_size,                      // batch size
                               const size_t sequence_length,                 // sequence length
                               const size_t past_buffer_sequence_length,     // sequence length in past state
                               const size_t present_buffer_sequence_length,  // sequence length in past state
                               const size_t head_size,                       // head size of Q, K, V
                               const size_t hidden_size,                     // hidden size of Output
                               const T* past_value,                          // past value only
                               T* present_value,                             // present value only
                               const bool past_present_share_buffer,         // whether present key and value share the same buffer
                               const bool packed_qkv,                        // whether Q, K, V are packed
                               const bool is_prompt,                         // whether it is prompt
                               ThreadPool* tp,
                               AllocatorPtr allocator) const {
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t kv_input_chunk_length = sequence_length * head_size;                     // L x H
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;        // L x H
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;  // T x H

    if (!past_present_share_buffer) {
      memset((void*)present_value,
             0,
             batch_size * kv_num_heads_ * present_buffer_sequence_length * head_size * sizeof(T));
    }

    const size_t loop_len = batch_size * num_heads_;

    // The cost of Gemm
    TensorOpCost unit_cost;
    unit_cost.compute_cycles =
        static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * present_buffer_sequence_length);
    unit_cost.bytes_loaded = static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + head_size) *
                                                 present_buffer_sequence_length * sizeof(T));
    unit_cost.bytes_stored = static_cast<double>(sequence_length * head_size * sizeof(T));

    if (present_value) {
      double bytes_to_copy_value = static_cast<double>(present_buff_chunk_length * sizeof(T));
      unit_cost.bytes_loaded += bytes_to_copy_value;
      unit_cost.bytes_stored += bytes_to_copy_value;
    }

    const size_t bytes_to_copy_trans = SafeInt<size_t>(head_size) * sizeof(T);
    double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
    unit_cost.bytes_loaded += bytes_to_copy_trans_all;
    unit_cost.bytes_stored += bytes_to_copy_trans_all;

    size_t output_fp32_bytes = 0;
    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      output_fp32_bytes = SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size * sizeof(float);
    }
    auto output_fp32 = allocator->Alloc(output_fp32_bytes);
    BufferUniquePtr scratch_buffer(output_fp32, BufferDeleter(allocator));

    ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const size_t batch_index = i / num_heads_;
        const size_t head_index = i % num_heads_;
        const size_t total_seqlen = static_cast<size_t>(seqlens_k[batch_index]) + 1;
        const size_t past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;  // Assume no padding sequence length
        const size_t past_chunk_length = past_seqlen * head_size;

        const T* v;
        if (packed_qkv) {
          v = V + packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor);
        } else {
          v = V + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        if (nullptr != present_value) {
          v = ConcatStateChunkGQA(past_value, v, present_value, present_buff_chunk_length, past_buff_chunk_length,
                                  past_chunk_length, kv_input_chunk_length, past_present_share_buffer,
                                  i / kv_num_heads_factor);
        }

        ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * present_buffer_sequence_length * i;

        if constexpr (std::is_same<T, float>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, /*alpha*/ attention_probs + attention_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v,
                                          static_cast<int>(head_size), 0.0f /*beta*/, output_current,
                                          static_cast<int>(hidden_size), nullptr);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          MlasGemm(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                   attention_probs + attention_probs_offset, static_cast<int>(present_buffer_sequence_length),
                   v, static_cast<int>(head_size), output_current, static_cast<int>(hidden_size),
                   MLFloat16(1.0f).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = head_size * total_seqlen * sizeof(float);
          auto v_fp32 = allocator->Alloc(bytes);
          BufferUniquePtr scratch_buffer(v_fp32, BufferDeleter(allocator));

          float* v_fp32_ptr = static_cast<float*>(v_fp32);
          MlasConvertHalfToFloatBuffer(v, v_fp32_ptr, head_size * total_seqlen);

          float* output_fp32_current = static_cast<float*>(output_fp32) +
                                       (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                                          1.f, /*alpha*/ attention_probs + attention_probs_offset,
                                          static_cast<int>(present_buffer_sequence_length), v_fp32_ptr,
                                          static_cast<int>(head_size), 0.0f /*beta*/, output_fp32_current,
                                          static_cast<int>(hidden_size), nullptr);
        }
      }
    });

    if constexpr (std::is_same<T, MLFloat16>::value && std::is_same<U, float>::value) {
      MlasConvertFloatToHalfBuffer(static_cast<float*>(output_fp32),
                                   output,
                                   SafeInt<size_t>(sequence_length) * batch_size * num_heads_ * head_size);
    }
  }

  template <typename T, typename U>
  void WriteOutputQKHeadChunk(T* output_qk, const U* attention_probs, size_t total_sequence_length) const {
    if (output_qk == nullptr) {
      return;
    }

    if constexpr (std::is_same_v<U, T>) {
      std::memcpy(output_qk, attention_probs, SafeInt<size_t>(total_sequence_length) * sizeof(T));
    } else {
      static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);
      MlasConvertFloatToHalfBuffer(static_cast<const float*>(attention_probs), output_qk, total_sequence_length);
    }
  }
};

}  // namespace contrib
}  // namespace onnxruntime
