// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <limits>
#include <string>
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas_qkv_quant.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime {
namespace contrib {

// Convert operator-level quantization attributes to the MLAS enum.
inline MLAS_KV_QUANT_TYPE ToMlasKVQuantType(KVQuantizationType quant_type, int bit_width) {
  if (bit_width == 8) {
    return quant_type == KVQuantizationType::PER_CHANNEL
               ? MLAS_KV_QUANT_TYPE::S8_PerChannel
               : MLAS_KV_QUANT_TYPE::S8_PerTensor;
  }
  return quant_type == KVQuantizationType::PER_CHANNEL
             ? MLAS_KV_QUANT_TYPE::S4_PerChannel
             : MLAS_KV_QUANT_TYPE::S4_PerTensor;
}

// GQA ConcatStateChunk variant for quantized KV cache.
// Copies past quantized bytes and quantizes new FP32 rows into the present buffer.
// Returns pointer to start of this head's slice in the present buffer.
inline const uint8_t* ConcatQuantStateChunkGQA(
    const uint8_t* past,
    const float* new_chunk,
    uint8_t* present,
    size_t present_buff_chunk_bytes,
    size_t past_buff_chunk_bytes,
    size_t past_chunk_bytes,
    size_t new_rows,
    size_t cols,
    size_t src_ld,
    MLAS_KV_QUANT_TYPE quant_type,
    const float* scales,
    bool past_present_share_buffer,
    std::ptrdiff_t kv_head_idx) {
  uint8_t* start = present + kv_head_idx * present_buff_chunk_bytes;
  uint8_t* p = start;

  if (!past_present_share_buffer && past_chunk_bytes > 0) {
    const uint8_t* src_past = past + kv_head_idx * past_buff_chunk_bytes;
    memcpy(p, src_past, past_chunk_bytes);
  }
  p += past_chunk_bytes;

  if (new_rows > 0) {
    MlasKVQuantize(new_chunk, p, new_rows, cols, src_ld, quant_type, scales, nullptr);
  }

  return start;
}

class GQAAttentionBase {
 protected:
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;

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

    k_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("k_quant_type", "NONE"));
    v_quant_type_ = StringToKVQuantizationType(info.GetAttrOrDefault<std::string>("v_quant_type", "NONE"));
    kv_cache_bit_width_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_cache_bit_width", 0));
    kv_quant_enabled_ = (k_quant_type_ != KVQuantizationType::NONE);

    disable_gqa_flash_ = ParseEnvironmentVariableWithDefault<bool>("ORT_GQA_DISABLE_FLASH_ATTENTION", false);

    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
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

  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  int kv_cache_bit_width_;
  bool kv_quant_enabled_;
  bool disable_gqa_flash_;

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
    const int kv_sequence_length = parameters.kv_sequence_length;
    const int total_sequence_length = parameters.total_sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = present_key != nullptr
                                      ? static_cast<int>(present_key->Shape().GetDims()[2])
                                      : parameters.total_sequence_length;

    // Shared KV: total_sequence_length must fit within the past buffer.
    if (kv_sequence_length == 0) {
      ORT_ENFORCE(total_sequence_length <= seqlen_past_kv_cache,
                  "total_seqlen (", total_sequence_length, ") exceeds past buffer size (",
                  seqlen_past_kv_cache, ") in shared KV mode");
    }

    // Compute the attention score.
    bool gqa_mlas_supported = MlasGQASupported<T>(CblasNoTrans, CblasTrans) &&
                              MlasGQASupported<T>(CblasNoTrans, CblasNoTrans);
    size_t bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * seqlen_present_kv_cache * (gqa_mlas_supported ? sizeof(T) : sizeof(float));
    auto attention_probs = allocator->Alloc(bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
    T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
    const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;
    T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

    const T* attention_bias_data = attention_bias != nullptr ? attention_bias->Data<T>() : nullptr;
    auto attention_bias_shape = attention_bias != nullptr ? attention_bias->Shape().GetDims() : gsl::span<const int64_t>{};

    bool past_present_share_buffer = past_key_data == present_key_data && past_value_data == present_value_data;

    const T* k = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;

    T* output_qk_buffer = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

    if (gqa_mlas_supported) {
      ComputeAttentionProbs(static_cast<T*>(attention_probs), Q, k, head_sink, seqlens_k->Data<int32_t>(), attention_bias_data,
                            batch_size, sequence_length, kv_sequence_length, total_sequence_length, attention_bias_shape, seqlen_past_kv_cache,
                            seqlen_present_kv_cache, head_size, past_key_data, present_key_data, output_qk_buffer,
                            past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);

      // Compute the attentionScore * Value: out(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
      const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
      ComputeVxAttentionScore(output->MutableData<T>(), static_cast<T*>(attention_probs), v,
                              seqlens_k->Data<int32_t>(),
                              batch_size, sequence_length, kv_sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                              hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                              is_prompt, tp, allocator);
    } else {
      ComputeAttentionProbs(static_cast<float*>(attention_probs), Q, k, head_sink, seqlens_k->Data<int32_t>(), attention_bias_data,
                            batch_size, sequence_length, kv_sequence_length, total_sequence_length, attention_bias_shape, seqlen_past_kv_cache,
                            seqlen_present_kv_cache, head_size, past_key_data, present_key_data, output_qk_buffer,
                            past_present_share_buffer, packed_qkv, is_prompt, tp, allocator);

      // Compute the attentionScore * Value: out(B, N, S, H_v) = attention_probs(B, N, S, T) x V(B, N, T, H_v)
      const T* v = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;
      ComputeVxAttentionScore(output->MutableData<T>(), static_cast<float*>(attention_probs), v,
                              seqlens_k->Data<int32_t>(),
                              batch_size, sequence_length, kv_sequence_length, seqlen_past_kv_cache, seqlen_present_kv_cache, head_size,
                              hidden_size, past_value_data, present_value_data, past_present_share_buffer, packed_qkv,
                              is_prompt, tp, allocator);
    }

    return Status::OK();
  }

  // Quantized KV cache attention path. Only supports T = float.
  // Uses MlasQKGemm / MlasSVGemm with quantized present K/V (uint8_t storage).
  Status ApplyAttentionQuantized(
      const float* Q,                // Q data [B, N, S, H] BNSH
      const float* K,                // K data [B, N_kv, L, H] or nullptr for packed_qkv
      const float* V,                // V data [B, N_kv, L, H] or nullptr for packed_qkv
      const float* head_sink,        // smooth softmax sink per head, or nullptr
      const Tensor* attention_bias,  // additive bias or nullptr
      const Tensor* past_key,        // past K (uint8_t)
      const Tensor* past_value,      // past V (uint8_t)
      Tensor* output,                // output [B, S, N*H] float
      Tensor* present_key,           // present K (uint8_t)
      Tensor* present_value,         // present V (uint8_t)
      Tensor* output_qk,
      const Tensor* seqlens_k,
      const float* k_scale,
      const float* v_scale,
      MLAS_KV_QUANT_TYPE quant_type,
      GroupQueryAttentionParameters& parameters,
      AllocatorPtr allocator,
      OpKernelContext* context) const {
    const bool is_prompt = parameters.is_first_prompt;
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int kv_sequence_length = parameters.kv_sequence_length;
    const int total_sequence_length = parameters.total_sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();
    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, head_size);

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = present_key != nullptr
                                      ? static_cast<int>(present_key->Shape().GetDims()[2])
                                      : parameters.total_sequence_length;

    if (kv_sequence_length == 0) {
      ORT_ENFORCE(total_sequence_length <= seqlen_past_kv_cache,
                  "total_seqlen (", total_sequence_length, ") exceeds past buffer size (",
                  seqlen_past_kv_cache, ") in shared KV mode");
    }

    // Allocate attention probs buffer (always float)
    size_t probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length *
                         seqlen_present_kv_cache * sizeof(float);
    auto attention_probs_alloc = allocator->Alloc(probs_bytes);
    BufferUniquePtr probs_buffer(attention_probs_alloc, BufferDeleter(allocator));
    float* attention_probs = static_cast<float*>(attention_probs_alloc);

    ORT_RETURN_IF(present_key == nullptr || present_value == nullptr,
                  "present_key and present_value must be provided for quantized KV cache");

    // Access cache data as raw bytes — INT4 uses uint8_t (packed nibbles),
    // INT8 uses int8_t. Both are accessed via uint8_t* for the MLAS quantize API.
    const uint8_t* past_key_data = nullptr;
    uint8_t* present_key_data = nullptr;
    const uint8_t* past_value_data = nullptr;
    uint8_t* present_value_data = nullptr;
    if (kv_cache_bit_width_ == 4) {
      past_key_data = past_key != nullptr ? past_key->Data<uint8_t>() : nullptr;
      present_key_data = present_key->MutableData<uint8_t>();
      past_value_data = past_value != nullptr ? past_value->Data<uint8_t>() : nullptr;
      present_value_data = present_value->MutableData<uint8_t>();
    } else {
      past_key_data = past_key != nullptr ? reinterpret_cast<const uint8_t*>(past_key->Data<int8_t>()) : nullptr;
      present_key_data = reinterpret_cast<uint8_t*>(present_key->MutableData<int8_t>());
      past_value_data = past_value != nullptr ? reinterpret_cast<const uint8_t*>(past_value->Data<int8_t>()) : nullptr;
      present_value_data = reinterpret_cast<uint8_t*>(present_value->MutableData<int8_t>());
    }

    const float* attention_bias_data = attention_bias != nullptr ? attention_bias->Data<float>() : nullptr;
    auto attention_bias_shape = attention_bias != nullptr
                                    ? attention_bias->Shape().GetDims()
                                    : gsl::span<const int64_t>{};

    bool past_present_share_buffer = (past_key_data == present_key_data) &&
                                     (past_value_data == present_value_data);

    const bool per_channel = (quant_type == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
                              quant_type == MLAS_KV_QUANT_TYPE::S4_PerChannel);

    const int32_t* seqlens_k_data = seqlens_k->Data<int32_t>();
    float* output_data = output->MutableData<float>();
    float* output_qk_buffer = output_qk != nullptr ? output_qk->MutableData<float>() : nullptr;

    // K/V base pointers (FP32, new tokens post-RoPE / from input)
    const float* k_base = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;
    const float* v_base = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;

    // Common loop parameters
    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_num_heads_factor = num_heads_ / kv_num_heads_;
    const size_t q_input_chunk_length = sequence_length * head_size;
    const size_t kv_input_chunk_length = kv_sequence_length * head_size;
    const size_t past_buff_chunk_bytes = SafeInt<size_t>(seqlen_past_kv_cache) * packed_row_bytes;
    const size_t present_buff_chunk_bytes = SafeInt<size_t>(seqlen_present_kv_cache) * packed_row_bytes;

    const size_t loop_len = batch_size * num_heads_;
    const float alpha = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    // ---- Concat K + QK^T + Softmax ----
    if (present_key_data && !past_present_share_buffer) {
      memset(present_key_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_bytes);
    }

    {
      TensorOpCost unit_cost;
      unit_cost.compute_cycles =
          static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * seqlen_present_kv_cache);
      // Q is FP32; K cache is packed (INT8 or INT4) bytes.
      unit_cost.bytes_loaded =
          static_cast<double>(sequence_length * head_size * sizeof(float) +
                              seqlen_present_kv_cache * packed_row_bytes);
      unit_cost.bytes_stored =
          static_cast<double>(SafeInt<ptrdiff_t>(sequence_length) * seqlen_present_kv_cache * sizeof(float));

      ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const size_t batch_index = i / num_heads_;
          const size_t head_index = i % num_heads_;
          const size_t total_seqlen = SafeInt<size_t>(seqlens_k_data[batch_index]) + 1;

          size_t past_seqlen, causal_past_seqlen;
          if (past_key == nullptr) {
            past_seqlen = 0;
            causal_past_seqlen = 0;
          } else if (kv_sequence_length == 0) {
            past_seqlen = total_seqlen;
            causal_past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;
          } else if (is_prompt) {
            past_seqlen = 0;
            causal_past_seqlen = 0;
          } else {
            past_seqlen = total_seqlen - sequence_length;
            causal_past_seqlen = past_seqlen;
          }

          const size_t kv_head_within_batch = head_index / kv_num_heads_factor;
          const std::ptrdiff_t kv_head_flat = static_cast<std::ptrdiff_t>(i / kv_num_heads_factor);
          const size_t past_chunk_bytes = past_seqlen * packed_row_bytes;

          // Concat quantized past K + quantize new K
          const float* k_new;
          if (packed_qkv) {
            k_new = k_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_within_batch;
          } else {
            k_new = k_base + kv_input_chunk_length * kv_head_flat;
          }
          const float* head_k_scale = per_channel
                                          ? k_scale + kv_head_within_batch * head_size
                                          : k_scale;

          const uint8_t* k_quantized = ConcatQuantStateChunkGQA(
              past_key_data, k_new, present_key_data,
              present_buff_chunk_bytes, past_buff_chunk_bytes,
              past_chunk_bytes, kv_sequence_length, head_size, head_size,
              quant_type, head_k_scale, past_present_share_buffer, kv_head_flat);

          // Q pointer
          const float* q;
          if (packed_qkv) {
            q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
          } else {
            q = Q + q_input_chunk_length * i;
          }

          // QK^T GEMM with quantized K cache
          const ptrdiff_t probs_offset =
              SafeInt<ptrdiff_t>(i) * sequence_length * seqlen_present_kv_cache;
          float* probs = attention_probs + probs_offset;

          MlasQKGemm(sequence_length, total_seqlen, head_size, alpha,
                     q, head_size, k_quantized, quant_type, head_k_scale,
                     probs, seqlen_present_kv_cache, nullptr);

          // Output QK buffer
          float* output_qk_thread = nullptr;
          if (output_qk_buffer != nullptr) {
            const ptrdiff_t output_qk_offset =
                SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length *
                (batch_index * num_heads_ + head_index);
            output_qk_thread = output_qk_buffer + output_qk_offset;
          }

          // Attention bias
          const float* attn_bias = nullptr;
          ptrdiff_t attn_bias_total_seqlen = 0;
          if (attention_bias_data != nullptr) {
            attn_bias_total_seqlen = static_cast<ptrdiff_t>(attention_bias_shape[3]);
            ptrdiff_t bias_offset = 0;
            const ptrdiff_t bias_matrix_size = sequence_length * attn_bias_total_seqlen;
            if (attention_bias_shape[0] != 1) {
              bias_offset += static_cast<ptrdiff_t>(
                  SafeInt<ptrdiff_t>(batch_index) * attention_bias_shape[1] *
                  bias_matrix_size);
            }
            if (attention_bias_shape[1] != 1) {
              bias_offset += SafeInt<ptrdiff_t>(head_index) * bias_matrix_size;
            }
            attn_bias = attention_bias_data + bias_offset;
          }

          // Softmax + masking (same as non-quantized path)
          float* sm = probs;
          for (size_t seq = 0; seq < static_cast<size_t>(sequence_length); seq++) {
            size_t seq_causal_length = causal_past_seqlen + seq + 1;

            // Cap effective causal length at total_seqlen so the softmax window stays within
            // the region filled by the QK GEMM. For right-padded batched prompts, padding
            // positions have seq_causal_length > total_seqlen; without this cap the softmax
            // would read uninitialized memory and produce NaN.
            const size_t effective_causal_length = std::min(seq_causal_length, total_seqlen);

            const bool apply_local = local_window_size_ >= 0 &&
                                     effective_causal_length > static_cast<size_t>(local_window_size_);
            const size_t start_off = apply_local ? effective_causal_length - local_window_size_ : 0;
            const size_t win_size = apply_local ? local_window_size_ : effective_causal_length;

            if (apply_local) {
              for (size_t t = 0; t < effective_causal_length - local_window_size_; t++) {
                sm[t] = 0.f;
              }
            }

            if (softcap_ > 0.f) {
              ComputeAttentionSoftcapInplace(sm + start_off, static_cast<int>(win_size), softcap_);
            }

            if (attn_bias != nullptr) {
              ApplyAttentionBias(sm + start_off, attn_bias + start_off, static_cast<int>(win_size));
            }

            for (size_t t = effective_causal_length; t < total_seqlen; t++) {
              sm[t] = 0.f;
            }

            if (qk_output_ == static_cast<int>(QKOutputType::BEFORE_SOFTMAX)) {
              WriteOutputQKHeadChunk<float, float>(output_qk_thread, sm, total_sequence_length);
            }

            if (use_smooth_softmax_ || head_sink != nullptr) {
              float sink = (head_sink != nullptr) ? head_sink[head_index] : 0.0f;
              ComputeSmoothSoftmaxInplace(sm + start_off, static_cast<int>(win_size), sink, nullptr);
            } else {
              ComputeAttentionSoftmaxInplace(sm + start_off, 1, static_cast<int>(win_size), nullptr);
            }

            if (qk_output_ == static_cast<int>(QKOutputType::AFTER_SOFTMAX)) {
              WriteOutputQKHeadChunk<float, float>(output_qk_thread, sm, total_sequence_length);
            }

            sm += seqlen_present_kv_cache;
            if (attn_bias != nullptr) {
              attn_bias += attn_bias_total_seqlen;
            }
            if (output_qk_thread != nullptr) {
              output_qk_thread += total_sequence_length;
            }
          }
        }
      });
    }

    // ---- Concat V + S*V ----
    if (!past_present_share_buffer) {
      memset(present_value_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_bytes);
    }

    {
      TensorOpCost unit_cost;
      unit_cost.compute_cycles =
          static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * head_size * seqlen_present_kv_cache);
      // Probs (softmax output) are FP32; V cache is packed (INT8 or INT4) bytes.
      unit_cost.bytes_loaded =
          static_cast<double>(SafeInt<ptrdiff_t>(sequence_length) * seqlen_present_kv_cache * sizeof(float) +
                              seqlen_present_kv_cache * packed_row_bytes);
      unit_cost.bytes_stored = static_cast<double>(sequence_length * head_size * sizeof(float));

      ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const size_t batch_index = i / num_heads_;
          const size_t head_index = i % num_heads_;
          const size_t total_seqlen = SafeInt<size_t>(seqlens_k_data[batch_index]) + 1;

          size_t past_seqlen;
          if (past_value == nullptr) {
            past_seqlen = 0;
          } else if (kv_sequence_length == 0) {
            past_seqlen = total_seqlen;
          } else if (is_prompt) {
            past_seqlen = 0;
          } else {
            past_seqlen = total_seqlen - sequence_length;
          }

          const size_t kv_head_within_batch = head_index / kv_num_heads_factor;
          const std::ptrdiff_t kv_head_flat = static_cast<std::ptrdiff_t>(i / kv_num_heads_factor);
          const size_t past_chunk_bytes = past_seqlen * packed_row_bytes;

          // Concat quantized past V + quantize new V
          const float* v_new;
          if (packed_qkv) {
            v_new = v_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_within_batch;
          } else {
            v_new = v_base + kv_input_chunk_length * kv_head_flat;
          }
          const float* head_v_scale = per_channel
                                          ? v_scale + kv_head_within_batch * head_size
                                          : v_scale;

          const uint8_t* v_quantized = ConcatQuantStateChunkGQA(
              past_value_data, v_new, present_value_data,
              present_buff_chunk_bytes, past_buff_chunk_bytes,
              past_chunk_bytes, kv_sequence_length, head_size, head_size,
              quant_type, head_v_scale, past_present_share_buffer, kv_head_flat);

          // S*V GEMM with quantized V cache
          ptrdiff_t probs_offset =
              SafeInt<ptrdiff_t>(sequence_length) * seqlen_present_kv_cache * i;
          float* output_current = output_data +
                                  (batch_index * sequence_length * num_heads_ + head_index) * head_size;

          MlasSVGemm(sequence_length, head_size, total_seqlen,
                     attention_probs + probs_offset, seqlen_present_kv_cache,
                     v_quantized, quant_type, head_v_scale,
                     output_current, hidden_size, 0.0f, nullptr);
        }
      });
    }

    return Status::OK();
  }

  // Flash Attention style tiled computation for quantized KV cache.
  // Avoids materializing the full [B, N, S, T] attention probability matrix.
  // Uses online softmax with KV block tiling for reduced memory usage.
  Status ApplyAttentionQuantizedFlash(
      const float* Q,                // Q data [B, N, S, H] BNSH
      const float* K,                // K data [B, N_kv, L, H] or nullptr for packed_qkv
      const float* V,                // V data [B, N_kv, L, H] or nullptr for packed_qkv
      const Tensor* attention_bias,  // additive bias [B|1, N|1, S, T] or nullptr
      const Tensor* past_key,        // past K (uint8_t)
      const Tensor* past_value,      // past V (uint8_t)
      Tensor* output,                // output [B, S, N*H] float
      Tensor* present_key,           // present K (uint8_t)
      Tensor* present_value,         // present V (uint8_t)
      const Tensor* seqlens_k,
      const float* k_scale,
      const float* v_scale,
      MLAS_KV_QUANT_TYPE quant_type,
      GroupQueryAttentionParameters& parameters,
      AllocatorPtr allocator,
      OpKernelContext* context) const {
    const bool is_prompt = parameters.is_first_prompt;
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int kv_sequence_length = parameters.kv_sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();
    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, head_size);

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = present_key != nullptr
                                      ? static_cast<int>(present_key->Shape().GetDims()[2])
                                      : parameters.total_sequence_length;

    if (kv_sequence_length == 0) {
      ORT_ENFORCE(parameters.total_sequence_length <= seqlen_past_kv_cache,
                  "total_seqlen (", parameters.total_sequence_length, ") exceeds past buffer size (",
                  seqlen_past_kv_cache, ") in shared KV mode");
    }

    ORT_RETURN_IF(present_key == nullptr || present_value == nullptr,
                  "present_key and present_value must be provided for quantized KV cache");

    // Access cache data as raw bytes
    const uint8_t* past_key_data = nullptr;
    uint8_t* present_key_data = nullptr;
    const uint8_t* past_value_data = nullptr;
    uint8_t* present_value_data = nullptr;
    if (kv_cache_bit_width_ == 4) {
      past_key_data = past_key != nullptr ? past_key->Data<uint8_t>() : nullptr;
      present_key_data = present_key->MutableData<uint8_t>();
      past_value_data = past_value != nullptr ? past_value->Data<uint8_t>() : nullptr;
      present_value_data = present_value->MutableData<uint8_t>();
    } else {
      past_key_data = past_key != nullptr ? reinterpret_cast<const uint8_t*>(past_key->Data<int8_t>()) : nullptr;
      present_key_data = reinterpret_cast<uint8_t*>(present_key->MutableData<int8_t>());
      past_value_data = past_value != nullptr ? reinterpret_cast<const uint8_t*>(past_value->Data<int8_t>()) : nullptr;
      present_value_data = reinterpret_cast<uint8_t*>(present_value->MutableData<int8_t>());
    }

    bool past_present_share_buffer = (past_key_data == present_key_data) &&
                                     (past_value_data == present_value_data);

    const bool per_channel = (quant_type == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
                              quant_type == MLAS_KV_QUANT_TYPE::S4_PerChannel);

    const int32_t* seqlens_k_data = seqlens_k->Data<int32_t>();

    // Attention bias setup
    const float* attention_bias_data = nullptr;
    int attention_bias_seqlen_stride = 0;
    bool attention_bias_broadcast_batch = true;
    bool attention_bias_broadcast_head = true;
    if (attention_bias != nullptr) {
      attention_bias_data = attention_bias->Data<float>();
      auto bias_shape = attention_bias->Shape().GetDims();
      attention_bias_seqlen_stride = static_cast<int>(bias_shape[3]);
      attention_bias_broadcast_batch = (bias_shape[0] == 1);
      attention_bias_broadcast_head = (bias_shape[1] == 1);
    }

    // K/V base pointers (FP32, new tokens)
    const float* k_base = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;
    const float* v_base = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;

    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_input_chunk_length = kv_sequence_length * head_size;
    const size_t past_buff_chunk_bytes = SafeInt<size_t>(seqlen_past_kv_cache) * packed_row_bytes;
    const size_t present_buff_chunk_bytes = SafeInt<size_t>(seqlen_present_kv_cache) * packed_row_bytes;

    // ---- Phase 1: Concat new K/V into present cache ----
    // We must do this first so the flash attention kernel can read the full present cache.
    if (present_key_data && !past_present_share_buffer) {
      memset(present_key_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_bytes);
      memset(present_value_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_bytes);
    }

    // Concat K and V caches (parallelize over batch * kv_num_heads)
    {
      const size_t concat_loop_len = batch_size * kv_num_heads_;
      TensorOpCost concat_cost;
      concat_cost.compute_cycles = static_cast<double>(kv_sequence_length * head_size);
      concat_cost.bytes_loaded = static_cast<double>(past_buff_chunk_bytes + kv_sequence_length * head_size * sizeof(float));
      concat_cost.bytes_stored = static_cast<double>(present_buff_chunk_bytes);

      ThreadPool::TryParallelFor(tp, concat_loop_len, concat_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t kv_idx = begin; kv_idx != end; ++kv_idx) {
          const size_t batch_index = kv_idx / kv_num_heads_;
          const size_t kv_head_index = kv_idx % kv_num_heads_;
          const size_t total_seqlen = SafeInt<size_t>(seqlens_k_data[batch_index]) + 1;

          size_t past_seqlen;
          if (past_key == nullptr) {
            past_seqlen = 0;
          } else if (kv_sequence_length == 0) {
            past_seqlen = total_seqlen;
          } else if (is_prompt) {
            past_seqlen = 0;
          } else {
            past_seqlen = total_seqlen - sequence_length;
          }
          const size_t past_chunk_bytes = past_seqlen * packed_row_bytes;

          const float* head_k_scale = per_channel
                                          ? k_scale + kv_head_index * head_size
                                          : k_scale;
          const float* head_v_scale = per_channel
                                          ? v_scale + kv_head_index * head_size
                                          : v_scale;

          // Concat K
          const float* k_new;
          if (packed_qkv) {
            k_new = k_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_index;
          } else {
            k_new = k_base + kv_input_chunk_length * kv_idx;
          }
          ConcatQuantStateChunkGQA(
              past_key_data, k_new, present_key_data,
              present_buff_chunk_bytes, past_buff_chunk_bytes,
              past_chunk_bytes, kv_sequence_length, head_size, head_size,
              quant_type, head_k_scale, past_present_share_buffer, kv_idx);

          // Concat V
          const float* v_new;
          if (packed_qkv) {
            v_new = v_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_index;
          } else {
            v_new = v_base + kv_input_chunk_length * kv_idx;
          }
          ConcatQuantStateChunkGQA(
              past_value_data, v_new, present_value_data,
              present_buff_chunk_bytes, past_buff_chunk_bytes,
              past_chunk_bytes, kv_sequence_length, head_size, head_size,
              quant_type, head_v_scale, past_present_share_buffer, kv_idx);
        }
      });
    }

    // ---- Phase 2: Flash Attention with quantized KV cache ----
    // Compute L2-aware block sizes (same formula as MHA flash attention)
    const auto& env = Env::Default();
    int l2_cache_size = env.GetL2CacheSize();

    // For quantized KV: effective bytes per KV element for cache considerations
    // We dequantize V blocks to FP32, so working set per KV row = head_size * sizeof(float)
    // K is accessed via MlasQKGemm which internally dequantizes; for block sizing purposes
    // treat it as FP32 working set.
    //
    // Working set in L2 per tile:
    //   Q slice: [Br, head_size] floats
    //   Scores:  [Br, Bc] floats
    //   V dequant: [Bc, head_size] floats
    //   Temp output: [Br, head_size] floats
    //   Total ~ (2*Br + Bc) * head_size + Br * Bc
    //   Approximation: use same formula as FP32 flash attention
    int kv_block_size = l2_cache_size / (static_cast<int>(sizeof(float)) * 4 * (head_size + head_size));
    kv_block_size = std::max(kv_block_size, 1);
    int q_block_size = std::min(kv_block_size, 2 * head_size);

    // The flash kernel uses a single (past_seqlen, total_seqlen) pair for all batch items.
    // When batch items have different seqlens_k (ragged), we must fall back to per-batch
    // invocation so each batch item gets its own correct causal offset.
    int max_total_seqlen = 0;
    int min_total_seqlen = std::numeric_limits<int>::max();
    int common_past_seqlen = 0;
    for (int b = 0; b < batch_size; ++b) {
      int total_sl = seqlens_k_data[b] + 1;
      max_total_seqlen = std::max(max_total_seqlen, total_sl);
      min_total_seqlen = std::min(min_total_seqlen, total_sl);
    }
    const bool ragged_seqlens = (max_total_seqlen != min_total_seqlen);

    if (ragged_seqlens) {
      // Ragged seqlens: each batch item has its own total_seqlen (and therefore
      // past_seqlen). Must use per-batch invocation regardless of past_key/prompt state.
      common_past_seqlen = -1;  // sentinel: per-batch
    } else if (past_key == nullptr || is_prompt) {
      common_past_seqlen = 0;
    } else if (kv_sequence_length == 0) {
      // Shared buffer mode: each batch item has its own past_seqlen.
      common_past_seqlen = -1;  // sentinel: per-batch
    } else {
      common_past_seqlen = max_total_seqlen - sequence_length;
    }

    // Cap block sizes
    kv_block_size = std::min(kv_block_size, max_total_seqlen);
    q_block_size = std::min(q_block_size, sequence_length);

    // Allocate per-thread buffers for flash attention
    int thread_count = concurrency::ThreadPool::DegreeOfParallelism(tp);
    thread_count = std::max(thread_count, 1);

    // Flash decoding: for decode (sequence_length==1), partition KV across threads
    // to improve parallelism when batch*heads < thread_count.
    const int kv_chunk_count = (max_total_seqlen + kv_block_size - 1) / kv_block_size;
    const bool use_flash_decoding = (sequence_length == 1 &&
                                     batch_size * num_heads_ < thread_count &&
                                     kv_chunk_count > 1);

    size_t buffer_size_per_thread;
    size_t partials_buffer_bytes = 0;
    if (use_flash_decoding) {
      // Flash decoding: per-thread scratch only needs scores[kv_block_size]
      buffer_size_per_thread = static_cast<size_t>(kv_block_size) * sizeof(float);
      // Partials: [batch * num_heads * kv_chunk_count * (2 + head_size)] floats
      partials_buffer_bytes = static_cast<size_t>(batch_size) * num_heads_ *
                              kv_chunk_count * (2 + head_size) * sizeof(float);
    } else {
      buffer_size_per_thread =
          (static_cast<size_t>(q_block_size) * 2 +                                   // l + m
           static_cast<size_t>(q_block_size) * static_cast<size_t>(kv_block_size) +  // scores
           static_cast<size_t>(q_block_size) * static_cast<size_t>(head_size)) *     // temp_output
          sizeof(float);
    }
    size_t total_buffer_bytes = buffer_size_per_thread * thread_count + partials_buffer_bytes;
    auto flash_buffer_alloc = allocator->Alloc(total_buffer_bytes);
    BufferUniquePtr flash_buffer(flash_buffer_alloc, BufferDeleter(allocator));

    // Partials buffer is placed after per-thread scratch
    float* partials_ptr = use_flash_decoding
                              ? reinterpret_cast<float*>(reinterpret_cast<char*>(flash_buffer_alloc) +
                                                         buffer_size_per_thread * thread_count)
                              : nullptr;

    // If all batch items share the same past_seqlen, use the unified flash kernel.
    // Otherwise, fall back to per-batch invocation.
    if (common_past_seqlen >= 0) {
      MlasFlashAttentionQuantizedKVArgs args;
      args.batch_size = batch_size;
      args.num_heads = num_heads_;
      args.kv_num_heads = kv_num_heads_;
      args.sequence_length = sequence_length;
      args.total_seqlen = max_total_seqlen;
      args.head_size = head_size;
      args.past_seqlen = common_past_seqlen;
      args.local_window_size = local_window_size_;
      args.seqlen_present_kv = seqlen_present_kv_cache;
      args.q_block_size = q_block_size;
      args.kv_block_size = kv_block_size;
      args.scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;
      args.quant_type = quant_type;
      args.per_channel_k = per_channel;
      args.per_channel_v = per_channel;
      args.thread_count = thread_count;
      args.buffer = reinterpret_cast<float*>(flash_buffer_alloc);
      args.buffer_size_per_thread = buffer_size_per_thread;
      args.query = Q;
      args.q_batch_stride = packed_qkv
                                ? static_cast<size_t>(packed_batch_stride)
                                : static_cast<size_t>(SafeInt<size_t>(num_heads_) * sequence_length * head_size);
      args.k_cache = present_key_data;
      args.v_cache = present_value_data;
      args.k_scale = k_scale;
      args.v_scale = v_scale;
      args.output = output->MutableData<float>();
      args.attention_bias = attention_bias_data;
      args.attention_bias_seqlen_stride = attention_bias_seqlen_stride;
      args.attention_bias_broadcast_batch = attention_bias_broadcast_batch;
      args.attention_bias_broadcast_head = attention_bias_broadcast_head;
      args.flash_decoding_partials = partials_ptr;
      args.kv_chunk_count = kv_chunk_count;

      MlasFlashAttentionQuantizedKV(&args, tp);
    } else {
      // Per-batch handling for variable past_seqlen (shared KV buffer mode or ragged seqlens)
      for (int b = 0; b < batch_size; ++b) {
        int total_sl = seqlens_k_data[b] + 1;
        // For prompt/no-past cases, past_seqlen is 0; otherwise derive from total_sl.
        int batch_past_seqlen = (past_key == nullptr || is_prompt)
                                    ? 0
                                    : std::max(0, total_sl - sequence_length);

        MlasFlashAttentionQuantizedKVArgs args;
        args.batch_size = 1;
        args.num_heads = num_heads_;
        args.kv_num_heads = kv_num_heads_;
        args.sequence_length = sequence_length;
        args.total_seqlen = total_sl;
        args.head_size = head_size;
        args.past_seqlen = batch_past_seqlen;
        args.local_window_size = local_window_size_;
        args.seqlen_present_kv = seqlen_present_kv_cache;
        args.q_block_size = q_block_size;
        args.kv_block_size = std::min(kv_block_size, total_sl);
        args.scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;
        args.quant_type = quant_type;
        args.per_channel_k = per_channel;
        args.per_channel_v = per_channel;
        args.thread_count = thread_count;
        args.buffer = reinterpret_cast<float*>(flash_buffer_alloc);
        args.buffer_size_per_thread = buffer_size_per_thread;

        // Offset Q and output for this batch
        const ptrdiff_t q_batch_stride_elems = packed_batch_stride > 0
                                                   ? packed_batch_stride
                                                   : static_cast<ptrdiff_t>(SafeInt<ptrdiff_t>(num_heads_) * sequence_length * head_size);
        args.query = Q + static_cast<size_t>(SafeInt<size_t>(b) * static_cast<size_t>(q_batch_stride_elems));
        args.q_batch_stride = static_cast<size_t>(q_batch_stride_elems);
        args.k_cache = present_key_data +
                       static_cast<size_t>(b) * kv_num_heads_ * seqlen_present_kv_cache * packed_row_bytes;
        args.v_cache = present_value_data +
                       static_cast<size_t>(b) * kv_num_heads_ * seqlen_present_kv_cache * packed_row_bytes;
        args.k_scale = k_scale;
        args.v_scale = v_scale;
        args.output = output->MutableData<float>() +
                      static_cast<size_t>(b) * sequence_length * hidden_size;

        // Slice attention bias for this batch (the kernel sees batch_size=1, so batch_idx=0 inside).
        // Bias shape is [batch|1, num_heads|1, S, T]; the batch stride uses the actual head
        // extent (1 when the head dim is broadcast).
        const float* batch_bias = attention_bias_data;
        if (attention_bias_data != nullptr && !attention_bias_broadcast_batch) {
          const size_t bias_head_extent = attention_bias_broadcast_head ? 1 : static_cast<size_t>(num_heads_);
          batch_bias += static_cast<size_t>(SafeInt<size_t>(b) * bias_head_extent * sequence_length * attention_bias_seqlen_stride);
        }
        args.attention_bias = batch_bias;
        args.attention_bias_seqlen_stride = attention_bias_seqlen_stride;
        args.attention_bias_broadcast_batch = true;  // batch offset handled above
        args.attention_bias_broadcast_head = attention_bias_broadcast_head;
        args.flash_decoding_partials = nullptr;  // per-batch doesn't use flash decoding
        args.kv_chunk_count = 0;

        MlasFlashAttentionQuantizedKV(&args, tp);
      }
    }

    return Status::OK();
  }

  // Non-quantized flash attention path. Only supports T = float.
  // Concatenates new K/V into the FP32 present cache, then runs the tiled
  // online-softmax kernel MlasFlashAttentionGQA (QK^T + softmax + S*V fused).
  Status ApplyAttentionFlash(
      const float* Q,                // Q data [B, N, S, H] BNSH
      const float* K,                // K data [B, N_kv, L, H] or nullptr for packed_qkv
      const float* V,                // V data [B, N_kv, L, H] or nullptr for packed_qkv
      const Tensor* attention_bias,  // additive bias [B|1, N|1, S, T] or nullptr
      const Tensor* past_key,        // past K (float)
      const Tensor* past_value,      // past V (float)
      Tensor* output,                // output [B, S, N*H] float
      Tensor* present_key,           // present K (float)
      Tensor* present_value,         // present V (float)
      const Tensor* seqlens_k,
      GroupQueryAttentionParameters& parameters,
      AllocatorPtr allocator,
      OpKernelContext* context) const {
    const bool is_prompt = parameters.is_first_prompt;
    const int batch_size = parameters.batch_size;
    const int sequence_length = parameters.sequence_length;
    const int kv_sequence_length = parameters.kv_sequence_length;
    const int head_size = parameters.head_size;
    const int hidden_size = parameters.hidden_size;
    const bool packed_qkv = parameters.is_packed_qkv;

    auto* tp = context->GetOperatorThreadPool();

    int seqlen_past_kv_cache = 0;
    if (past_key != nullptr && past_value != nullptr) {
      seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
    }
    int seqlen_present_kv_cache = present_key != nullptr
                                      ? static_cast<int>(present_key->Shape().GetDims()[2])
                                      : parameters.total_sequence_length;

    if (kv_sequence_length == 0) {
      ORT_ENFORCE(parameters.total_sequence_length <= seqlen_past_kv_cache,
                  "total_seqlen (", parameters.total_sequence_length, ") exceeds past buffer size (",
                  seqlen_past_kv_cache, ") in shared KV mode");
    }

    ORT_RETURN_IF(present_key == nullptr || present_value == nullptr,
                  "present_key and present_value must be provided for flash attention");

    const float* past_key_data = past_key != nullptr ? past_key->Data<float>() : nullptr;
    float* present_key_data = present_key->MutableData<float>();
    const float* past_value_data = past_value != nullptr ? past_value->Data<float>() : nullptr;
    float* present_value_data = present_value->MutableData<float>();

    bool past_present_share_buffer = (past_key_data == present_key_data) &&
                                     (past_value_data == present_value_data);

    const int32_t* seqlens_k_data = seqlens_k->Data<int32_t>();

    // Attention bias setup
    const float* attention_bias_data = nullptr;
    int attention_bias_seqlen_stride = 0;
    bool attention_bias_broadcast_batch = true;
    bool attention_bias_broadcast_head = true;
    if (attention_bias != nullptr) {
      attention_bias_data = attention_bias->Data<float>();
      auto bias_shape = attention_bias->Shape().GetDims();
      attention_bias_seqlen_stride = static_cast<int>(bias_shape[3]);
      attention_bias_broadcast_batch = (bias_shape[0] == 1);
      attention_bias_broadcast_head = (bias_shape[1] == 1);
    }

    // K/V base pointers (FP32, new tokens)
    const float* k_base = packed_qkv ? Q + num_heads_ * sequence_length * head_size : K;
    const float* v_base = packed_qkv ? Q + (num_heads_ + kv_num_heads_) * sequence_length * head_size : V;

    const ptrdiff_t packed_batch_stride =
        packed_qkv ? SafeInt<ptrdiff_t>(num_heads_ + 2 * kv_num_heads_) * sequence_length * head_size
                   : SafeInt<ptrdiff_t>(0);
    const size_t kv_input_chunk_length = SafeInt<size_t>(kv_sequence_length) * head_size;
    const size_t past_buff_chunk_length = SafeInt<size_t>(seqlen_past_kv_cache) * head_size;
    const size_t present_buff_chunk_length = SafeInt<size_t>(seqlen_present_kv_cache) * head_size;

    // ---- Phase 1: Concat new K/V into present cache ----
    // We must do this first so the flash attention kernel can read the full present cache.
    if (present_key_data && !past_present_share_buffer) {
      memset(present_key_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_length * sizeof(float));
      memset(present_value_data, 0,
             SafeInt<size_t>(batch_size) * kv_num_heads_ * present_buff_chunk_length * sizeof(float));
    }

    // Concat K and V caches (parallelize over batch * kv_num_heads)
    {
      const size_t concat_loop_len = batch_size * kv_num_heads_;
      TensorOpCost concat_cost;
      concat_cost.compute_cycles = static_cast<double>(kv_sequence_length * head_size);
      concat_cost.bytes_loaded = static_cast<double>((past_buff_chunk_length + kv_input_chunk_length) * sizeof(float));
      concat_cost.bytes_stored = static_cast<double>(present_buff_chunk_length * sizeof(float));

      ThreadPool::TryParallelFor(tp, concat_loop_len, concat_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t kv_idx = begin; kv_idx != end; ++kv_idx) {
          const size_t batch_index = kv_idx / kv_num_heads_;
          const size_t kv_head_index = kv_idx % kv_num_heads_;
          const size_t total_seqlen = SafeInt<size_t>(seqlens_k_data[batch_index]) + 1;

          size_t past_seqlen;
          if (past_key == nullptr) {
            past_seqlen = 0;
          } else if (kv_sequence_length == 0) {
            past_seqlen = total_seqlen;
          } else if (is_prompt) {
            past_seqlen = 0;
          } else {
            past_seqlen = total_seqlen - sequence_length;
          }
          const size_t past_chunk_length = past_seqlen * head_size;

          // Concat K
          const float* k_new;
          if (packed_qkv) {
            k_new = k_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_index;
          } else {
            k_new = k_base + kv_input_chunk_length * kv_idx;
          }
          ConcatStateChunkGQA(past_key_data, k_new, present_key_data,
                              present_buff_chunk_length, past_buff_chunk_length,
                              past_chunk_length, kv_input_chunk_length,
                              past_present_share_buffer, kv_idx);

          // Concat V
          const float* v_new;
          if (packed_qkv) {
            v_new = v_base + packed_batch_stride * batch_index +
                    kv_input_chunk_length * kv_head_index;
          } else {
            v_new = v_base + kv_input_chunk_length * kv_idx;
          }
          ConcatStateChunkGQA(past_value_data, v_new, present_value_data,
                              present_buff_chunk_length, past_buff_chunk_length,
                              past_chunk_length, kv_input_chunk_length,
                              past_present_share_buffer, kv_idx);
        }
      });
    }

    // ---- Phase 2: Flash Attention with FP32 KV cache ----
    // Compute L2-aware block sizes (same formula as MHA flash attention).
    const auto& env = Env::Default();
    int l2_cache_size = env.GetL2CacheSize();

    int kv_block_size = l2_cache_size / (static_cast<int>(sizeof(float)) * 4 * (head_size + head_size));
    kv_block_size = std::max(kv_block_size, 1);
    int q_block_size = std::min(kv_block_size, 2 * head_size);

    // The flash kernel uses a single (past_seqlen, total_seqlen) pair for all batch items.
    // When batch items have different seqlens_k (ragged), fall back to per-batch invocation
    // so each batch item gets its own correct causal offset.
    int max_total_seqlen = 0;
    int min_total_seqlen = std::numeric_limits<int>::max();
    int common_past_seqlen = 0;
    for (int b = 0; b < batch_size; ++b) {
      int total_sl = seqlens_k_data[b] + 1;
      max_total_seqlen = std::max(max_total_seqlen, total_sl);
      min_total_seqlen = std::min(min_total_seqlen, total_sl);
    }
    const bool ragged_seqlens = (max_total_seqlen != min_total_seqlen);

    if (ragged_seqlens) {
      common_past_seqlen = -1;  // sentinel: per-batch
    } else if (past_key == nullptr || is_prompt) {
      common_past_seqlen = 0;
    } else if (kv_sequence_length == 0) {
      // Shared buffer mode: each batch item has its own past_seqlen.
      common_past_seqlen = -1;  // sentinel: per-batch
    } else {
      common_past_seqlen = max_total_seqlen - sequence_length;
    }

    // Cap block sizes
    kv_block_size = std::min(kv_block_size, max_total_seqlen);
    q_block_size = std::min(q_block_size, sequence_length);

    int thread_count = concurrency::ThreadPool::DegreeOfParallelism(tp);
    thread_count = std::max(thread_count, 1);

    // Flash decoding: for decode (sequence_length==1), partition KV across threads
    // to improve parallelism when batch*heads < thread_count. This KV-split is only
    // wired into the unified kernel (common_past_seqlen >= 0); the ragged/per-batch
    // fallback runs the single-pass decode kernel instead, which needs a larger
    // per-thread scratch (scores[total_seqlen] + temp_output[head_size]). Gating on
    // common_past_seqlen >= 0 keeps the per-thread buffer sizing below consistent
    // with the kernel that actually runs.
    const int kv_chunk_count = (max_total_seqlen + kv_block_size - 1) / kv_block_size;
    const bool use_flash_decoding = (sequence_length == 1 &&
                                     common_past_seqlen >= 0 &&
                                     batch_size * num_heads_ < thread_count &&
                                     kv_chunk_count > 1);

    size_t buffer_size_per_thread;
    size_t partials_buffer_bytes = 0;
    if (use_flash_decoding) {
      // Flash decoding: per-thread scratch only needs scores[kv_block_size]
      buffer_size_per_thread = SafeInt<size_t>(kv_block_size) * sizeof(float);
      // Partials: [batch * num_heads * kv_chunk_count * (2 + head_size)] floats
      partials_buffer_bytes = SafeInt<size_t>(batch_size) * num_heads_ *
                              kv_chunk_count * (2 + head_size) * sizeof(float);
    } else if (sequence_length == 1) {
      // Decode (GEMV kernel, no Q/KV tiling): per-thread scratch holds the full
      // score row scores[total_seqlen] plus a temp output accumulator[head_size].
      buffer_size_per_thread =
          (SafeInt<size_t>(max_total_seqlen) + head_size) * sizeof(float);
    } else {
      buffer_size_per_thread =
          (SafeInt<size_t>(q_block_size) * 2 +              // l + m
           SafeInt<size_t>(q_block_size) * kv_block_size +  // scores
           SafeInt<size_t>(q_block_size) * head_size) *     // temp_output
          sizeof(float);
    }
    size_t total_buffer_bytes = SafeInt<size_t>(buffer_size_per_thread) * thread_count + partials_buffer_bytes;
    auto flash_buffer_alloc = allocator->Alloc(total_buffer_bytes);
    BufferUniquePtr flash_buffer(flash_buffer_alloc, BufferDeleter(allocator));

    // Partials buffer is placed after per-thread scratch
    float* partials_ptr = use_flash_decoding
                              ? reinterpret_cast<float*>(reinterpret_cast<char*>(flash_buffer_alloc) +
                                                         buffer_size_per_thread * thread_count)
                              : nullptr;

    const float scale = scale_ == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale_;

    // If all batch items share the same past_seqlen, use the unified flash kernel.
    // Otherwise, fall back to per-batch invocation.
    if (common_past_seqlen >= 0) {
      MlasFlashAttentionGQAArgs args;
      args.batch_size = batch_size;
      args.num_heads = num_heads_;
      args.kv_num_heads = kv_num_heads_;
      args.sequence_length = sequence_length;
      args.total_seqlen = max_total_seqlen;
      args.head_size = head_size;
      args.past_seqlen = common_past_seqlen;
      args.local_window_size = local_window_size_;
      args.seqlen_present_kv = seqlen_present_kv_cache;
      args.q_block_size = q_block_size;
      args.kv_block_size = kv_block_size;
      args.scale = scale;
      args.thread_count = thread_count;
      args.buffer = reinterpret_cast<float*>(flash_buffer_alloc);
      args.buffer_size_per_thread = buffer_size_per_thread;
      args.query = Q;
      args.q_batch_stride = packed_qkv
                                ? static_cast<size_t>(packed_batch_stride)
                                : static_cast<size_t>(SafeInt<size_t>(num_heads_) * sequence_length * head_size);
      args.k_cache = present_key_data;
      args.v_cache = present_value_data;
      args.output = output->MutableData<float>();
      args.attention_bias = attention_bias_data;
      args.attention_bias_seqlen_stride = attention_bias_seqlen_stride;
      args.attention_bias_broadcast_batch = attention_bias_broadcast_batch;
      args.attention_bias_broadcast_head = attention_bias_broadcast_head;
      args.flash_decoding_partials = partials_ptr;
      args.kv_chunk_count = kv_chunk_count;

      MlasFlashAttentionGQA(&args, tp);
    } else {
      // Per-batch handling for variable past_seqlen (shared KV buffer mode or ragged seqlens)
      for (int b = 0; b < batch_size; ++b) {
        int total_sl = seqlens_k_data[b] + 1;
        int batch_past_seqlen = (past_key == nullptr || is_prompt)
                                    ? 0
                                    : std::max(0, total_sl - sequence_length);

        MlasFlashAttentionGQAArgs args;
        args.batch_size = 1;
        args.num_heads = num_heads_;
        args.kv_num_heads = kv_num_heads_;
        args.sequence_length = sequence_length;
        args.total_seqlen = total_sl;
        args.head_size = head_size;
        args.past_seqlen = batch_past_seqlen;
        args.local_window_size = local_window_size_;
        args.seqlen_present_kv = seqlen_present_kv_cache;
        args.q_block_size = q_block_size;
        args.kv_block_size = std::min(kv_block_size, total_sl);
        args.scale = scale;
        args.thread_count = thread_count;
        args.buffer = reinterpret_cast<float*>(flash_buffer_alloc);
        args.buffer_size_per_thread = buffer_size_per_thread;

        // Offset Q and output for this batch
        const ptrdiff_t q_batch_stride_elems = packed_batch_stride > 0
                                                   ? packed_batch_stride
                                                   : static_cast<ptrdiff_t>(SafeInt<ptrdiff_t>(num_heads_) * sequence_length * head_size);
        args.query = Q + static_cast<size_t>(b) * static_cast<size_t>(q_batch_stride_elems);
        args.q_batch_stride = SafeInt<size_t>(num_heads_) * sequence_length * head_size;
        args.k_cache = present_key_data +
                       static_cast<size_t>(b) * kv_num_heads_ * present_buff_chunk_length;
        args.v_cache = present_value_data +
                       static_cast<size_t>(b) * kv_num_heads_ * present_buff_chunk_length;
        args.output = output->MutableData<float>() +
                      static_cast<size_t>(b) * sequence_length * hidden_size;

        // Slice attention bias for this batch (the kernel sees batch_size=1, so batch_idx=0 inside).
        // Bias shape is [batch|1, num_heads|1, S, T]; the batch stride uses the actual head
        // extent (1 when the head dim is broadcast).
        const float* batch_bias = attention_bias_data;
        if (attention_bias_data != nullptr && !attention_bias_broadcast_batch) {
          const size_t bias_head_extent = attention_bias_broadcast_head ? 1 : static_cast<size_t>(num_heads_);
          batch_bias += static_cast<size_t>(b) * bias_head_extent * sequence_length * attention_bias_seqlen_stride;
        }
        args.attention_bias = batch_bias;
        args.attention_bias_seqlen_stride = attention_bias_seqlen_stride;
        args.attention_bias_broadcast_batch = true;  // batch offset handled above
        args.attention_bias_broadcast_head = attention_bias_broadcast_head;
        args.flash_decoding_partials = nullptr;  // per-batch doesn't use flash decoding
        args.kv_chunk_count = 0;

        MlasFlashAttentionGQA(&args, tp);
      }
    }

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
  //  attention_probs(B, N, S, T) = Softmax(attention_probs)
  // If T is float32, U is float32. If T is float16, U could be float16 or float32.
  template <typename T, typename U>
  void ComputeAttentionProbs(U* attention_probs,                                   // output probs [B, N, S, T]
                             const T* Q,                                           // query [B, N, S, H] (BNSH)
                             const T* K,                                           // key input [B, N_kv, L, H] (BNSH); L=0 for shared KV
                             const T* head_sink,                                   // smooth softmax sink per head, or nullptr
                             const int32_t* seqlens_k,                             // total_sequence_length - 1 per batch
                             const T* attention_bias,                              // additive bias [B|1, N|1, S, T], or nullptr
                             const size_t batch_size,                              // batch size
                             const size_t sequence_length,                         // Q sequence length (new tokens)
                             const size_t kv_sequence_length,                      // K/V input sequence length; 0 for shared KV
                             const size_t total_sequence_length,                   // total tokens (past + new)
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
    const size_t q_input_chunk_length = sequence_length * head_size;
    const size_t kv_input_chunk_length = kv_sequence_length * head_size;
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;

    if (present_key && !past_present_share_buffer) {
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
        const size_t total_seqlen = SafeInt<size_t>(seqlens_k[batch_index]) + 1;
        // past_seqlen: how much data to copy from past buffer in ConcatStateChunkGQA.
        // causal_past_seqlen: offset for causal masking (seq_causal_length = causal_past_seqlen + seq + 1).
        // These differ for shared KV prompt: copy all past data, but causal starts at 0.
        size_t past_seqlen;
        size_t causal_past_seqlen;
        if (past_key == nullptr) {
          past_seqlen = 0;
          causal_past_seqlen = 0;
        } else if (kv_sequence_length == 0) {
          past_seqlen = total_seqlen;  // Copy all KV data from past (shared KV)
          causal_past_seqlen = is_prompt ? 0 : total_seqlen - sequence_length;
        } else if (is_prompt) {
          past_seqlen = 0;
          causal_past_seqlen = 0;
        } else {
          past_seqlen = total_seqlen - sequence_length;
          causal_past_seqlen = past_seqlen;
        }
        const size_t past_chunk_length = SafeInt<size_t>(past_seqlen) * head_size;

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
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr, &mlas_backend_kernel_selector_config_);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          MlasGemm(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size,
                   q, static_cast<int>(head_size), k, static_cast<int>(head_size), output,
                   static_cast<int>(present_buffer_sequence_length),
                   MLFloat16(alpha).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = SafeInt<size_t>(head_size) * (sequence_length + total_seqlen) * sizeof(float);
          auto q_k_fp32 = allocator->Alloc(bytes);
          BufferUniquePtr scratch_buffer(q_k_fp32, BufferDeleter(allocator));

          float* q_fp32 = static_cast<float*>(q_k_fp32);
          MlasConvertHalfToFloatBuffer(q, q_fp32, head_size * sequence_length);

          float* k_fp32 = q_fp32 + head_size * sequence_length;
          MlasConvertHalfToFloatBuffer(k, k_fp32, head_size * total_seqlen);

          math::GemmEx<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, total_seqlen, head_size, alpha, q_fp32,
                                          static_cast<int>(head_size), k_fp32, static_cast<int>(head_size), 0.0f /*bata*/,
                                          output, static_cast<int>(present_buffer_sequence_length), nullptr, &mlas_backend_kernel_selector_config_);
        }

        // Pre-allocate buffer for attention mask to avoid allocating it for every processed token
        float* attention_bias_thread_fp32 = nullptr;
        if (attention_bias_thread != nullptr) {
          if constexpr (!std::is_same_v<U, T>) {
            static_assert(std::is_same_v<U, float> && std::is_same_v<T, MLFloat16>);

            size_t bytes = SafeInt<size_t>(attention_total_seqlen) * sizeof(float);
            attention_bias_thread_fp32 = static_cast<float*>(allocator->Alloc(bytes));
          }
        }
        BufferUniquePtr scratch_buffer(attention_bias_thread_fp32, BufferDeleter(allocator));

        // compute Softmax
        U* output_softmax = output;
        for (size_t seq = 0; seq < sequence_length; seq++) {
          size_t seq_causal_length = causal_past_seqlen + seq + 1;

          // For right-padded batched prompts, padding positions have seq_causal_length > total_seqlen.
          // The GEMM only fills columns [0, total_seqlen); beyond that the buffer is uninitialized.
          // Cap the effective causal length so the softmax window stays within the filled region,
          // preventing NaN from uninitialized memory propagating into the output.
          const size_t effective_causal_length = std::min(seq_causal_length, total_seqlen);

          const bool should_apply_local_window = local_window_size_ >= 0 &&
                                                 effective_causal_length > static_cast<size_t>(local_window_size_);

          const size_t start_offset = should_apply_local_window ? effective_causal_length - local_window_size_ : 0;
          const size_t window_size = should_apply_local_window ? local_window_size_ : effective_causal_length;

          // Mask everything before local window, if local window should be applied
          if (should_apply_local_window) {
            for (size_t total_seq_id = 0; total_seq_id < effective_causal_length - local_window_size_; total_seq_id++) {
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

          // set causal [effective_causal_length, total_seqlen) to 0.f
          for (size_t total_seq_id = effective_causal_length; total_seq_id < total_seqlen; total_seq_id++) {
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
                               const size_t sequence_length,                 // sequence length of Q
                               const size_t kv_sequence_length,              // sequence length of K/V input
                               const size_t past_buffer_sequence_length,     // sequence length in past state
                               const size_t present_buffer_sequence_length,  // sequence length in present state
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
    const size_t kv_input_chunk_length = kv_sequence_length * head_size;
    const size_t past_buff_chunk_length = past_buffer_sequence_length * head_size;
    const size_t present_buff_chunk_length = present_buffer_sequence_length * head_size;

    if (present_value && !past_present_share_buffer) {
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
        const size_t total_seqlen = SafeInt<size_t>(seqlens_k[batch_index]) + 1;
        size_t past_seqlen;
        if (past_value == nullptr) {
          past_seqlen = 0;
        } else if (kv_sequence_length == 0) {
          past_seqlen = total_seqlen;
        } else if (is_prompt) {
          past_seqlen = 0;
        } else {
          past_seqlen = total_seqlen - sequence_length;
        }
        const size_t past_chunk_length = SafeInt<size_t>(past_seqlen) * head_size;

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
                                          static_cast<int>(hidden_size), nullptr, &mlas_backend_kernel_selector_config_);
        } else if constexpr (std::is_same<U, MLFloat16>::value) {
          T* output_current = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
          MlasGemm(CblasNoTrans, CblasNoTrans, sequence_length, head_size, total_seqlen,
                   attention_probs + attention_probs_offset, static_cast<int>(present_buffer_sequence_length),
                   v, static_cast<int>(head_size), output_current, static_cast<int>(hidden_size),
                   MLFloat16(1.0f).val, static_cast<uint16_t>(0) /*beta*/, nullptr);
        } else {
          size_t bytes = SafeInt<size_t>(head_size) * total_seqlen * sizeof(float);
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
                                          static_cast<int>(hidden_size), nullptr, &mlas_backend_kernel_selector_config_);
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
