// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace multihead_attention_helper {

template <typename T>
Status Check_QKV(const T* packed_qkv, AttentionQkvFormat& qkv_format) {
  const auto& query_dims = packed_qkv->Shape().GetDims();
  if (query_dims.size() == 3) {
    // Packed qkv used by DecoderMaskedMultiHeadAttention. Query shape is (B, S, 3D), no key and value.
    qkv_format = AttentionQkvFormat::QKV_BS3NH;
  } else {
    assert(query_dims.size() == 5);
    if (static_cast<int>(query_dims[3]) != 3) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Expect 'query' shape (batch_size, sequence_length, num_heads, 3, head_size) for packed qkv");
    }

    qkv_format = AttentionQkvFormat::QKV_BSN3H;
  }

  return Status::OK();
}

template <typename T>
Status Check_Q_KV(const T* query, const T* packed_kv, int num_heads, int head_size,
                  AttentionQkvFormat& qkv_format, int& kv_sequence_length) {
  const auto& query_dims = query->Shape().GetDims();
  const auto& key_dims = packed_kv->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expect rank of query be 3 for packed kv");
  }

  if (key_dims.size() != 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expect rank of key be 5 for packed kv");
  }

  if (key_dims[0] != query_dims[0] ||
      static_cast<int>(key_dims[2]) != num_heads ||
      static_cast<int>(key_dims[3]) != 2 ||
      static_cast<int>(key_dims[4]) != head_size) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Expect 'key' shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv");
  }

  qkv_format = AttentionQkvFormat::Q_KV_BSNH_BSN2H;
  kv_sequence_length = static_cast<int>(key_dims[1]);
  return Status::OK();
}

template <typename T>
Status Check_Q_K_V(const T* query, const T* key, const T* value, int num_heads, int head_size,
                   AttentionQkvFormat& qkv_format, int& kv_sequence_length, int& v_hidden_size) {
  const auto& query_dims = query->Shape().GetDims();
  const auto& key_dims = key->Shape().GetDims();
  const auto& value_dims = value->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expect rank of query be 3 for packed kv");
  }

  if (key_dims.size() != value_dims.size() || (key_dims.size() != 3 && value_dims.size() != 4)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Expect rank of key and value be same, and either 3 or 4");
  }

  if (key_dims[0] != query_dims[0] || value_dims[0] != query_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query', 'key' and 'value' shall have same dim 0 (batch_size)");
  }

  if (key_dims.size() == 3) {
    if (key_dims[2] != query_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'key' shall have same dim 2 (hidden_size)");
    }

    if (key_dims[1] != value_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' and 'value' shall have same dim 1 (kv_sequence_length)");
    }

    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
    kv_sequence_length = static_cast<int>(key_dims[1]);
    v_hidden_size = static_cast<int>(value_dims[2]);
  } else {  // key_dims.size() == 4
    if (value->Shape() != key->Shape() ||
        static_cast<int>(key_dims[1]) != num_heads ||
        static_cast<int>(key_dims[3]) != head_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'key' and 'value' shall have same shape (batch_size, num_heads, kv_sequence_length, head_size)");
    }

    qkv_format = AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH;
    kv_sequence_length = static_cast<int>(key_dims[2]);
    v_hidden_size = static_cast<int>(value_dims[1]) * static_cast<int>(value_dims[3]);
  }

  return Status::OK();
}

template <typename T>
Status CheckPast(const T* past_key, const T* past_value, const T* past_seq_len,
                 int batch_size, int num_heads, int head_size, bool past_present_share_buffer,
                 int& past_sequence_length, int& max_sequence_length) {
  const auto& past_key_dims = past_key->Shape().GetDims();
  const auto& past_value_dims = past_value->Shape().GetDims();

  if (past_key_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' is expected to have 4 dimensions, got ",
                           past_key_dims.size());
  }
  if (past_value_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_value' is expected to have 4 dimensions, got ",
                           past_value_dims.size());
  }

  if (past_key_dims[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' dimension 0 should be batch_size, got ",
                           past_key_dims[0]);
  }
  if (past_value_dims[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_value' dimension 0 should be batch_size, got ",
                           past_value_dims[0]);
  }

  if (past_key_dims[1] != num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' dimension 1 should be same as number of heads, got ",
                           past_key_dims[1]);
  }
  if (past_value_dims[1] != num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_value' dimension 1 should be same as number of heads, got ",
                           past_value_dims[1]);
  }
  if (past_key_dims[2] != past_value_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' and 'past_value' shall have same dim 2 (past_sequence_length). ",
                           past_key_dims[2], " vs ", past_value_dims[2]);
  }
  if (past_key_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' dimension 3 should be same as head_size, got ",
                           past_key_dims[3]);
  }
  if (past_value_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_value' dimension 3 should be same as head_size, got ",
                           past_value_dims[3]);
  }
  past_sequence_length = static_cast<int>(past_key_dims[2]);
  if (past_present_share_buffer) {
    max_sequence_length = static_cast<int>(past_key_dims[2]);
    if (past_seq_len == nullptr || !onnxruntime::IsScalarOr1ElementVector(past_seq_len)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "past_sequence_length tensor must be of one element when past_present_share_buffer is set");
    }
    past_sequence_length = *((*past_seq_len).template Data<int32_t>());
  }
  return Status::OK();
}

inline Status CheckAttentionBias(
    const gsl::span<const int64_t>& attention_bias_dims,
    int64_t batch_size, int64_t num_heads, int64_t sequence_length, int64_t total_sequence_length) {
  if (attention_bias_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_bias' is expected to have 4 dimensions, got ",
                           attention_bias_dims.size());
  }

  if (attention_bias_dims[0] != batch_size && attention_bias_dims[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_bias' dimension 0 should be batch_size or 1, got ",
                           attention_bias_dims[0]);
  }

  if (attention_bias_dims[1] != num_heads && attention_bias_dims[1] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_bias' dimension 1 should be same as number of heads or 1, got ",
                           attention_bias_dims[1]);
  }
  if (attention_bias_dims[2] != sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_bias' dimension 2 should be same as sequence_length, got ",
                           attention_bias_dims[2]);
  }
  if (attention_bias_dims[3] != total_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_bias' dimension 3 should be same as total_sequence_length, got ",
                           attention_bias_dims[3]);
  }
  return Status::OK();
}

template <typename T>
AttentionMaskType GetMaskType(const T* key_padding_mask, int batch_size, int sequence_length, int total_sequence_length) {
  AttentionMaskType mask_type = AttentionMaskType::MASK_UNKNOWN;
  const auto& mask_dims = key_padding_mask->Shape().GetDims();
  if (mask_dims.size() == 1) {
    if (mask_dims[0] == static_cast<int64_t>(batch_size)) {
      mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
    } else if (mask_dims[0] == static_cast<int64_t>(3) * static_cast<int64_t>(batch_size) + static_cast<int64_t>(2)) {
      mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START;
    }
  } else if (mask_dims.size() == 2 && mask_dims[0] == static_cast<int64_t>(batch_size) &&
             mask_dims[1] == static_cast<int64_t>(total_sequence_length)) {
    mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
  } else if (mask_dims.size() == 3 && mask_dims[0] == static_cast<int64_t>(batch_size) &&
             mask_dims[1] == static_cast<int64_t>(sequence_length) &&
             mask_dims[2] == static_cast<int64_t>(total_sequence_length)) {
    mask_type = AttentionMaskType::MASK_3D_ATTENTION;
  }
  return mask_type;
}

template <typename T>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* bias,
                   const T* key_padding_mask,
                   const T* attention_bias,
                   const T* past_key,
                   const T* past_value,
                   const T* past_seq_len,
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   float scale,
                   bool is_unidirectional,
                   bool past_present_share_buffer,
                   AttentionType operator_type) {
  // ---------------------------------------------------------------
  // Notations:
  //    B: batch_size
  //    N: num_heads
  //    H: head_size of Q and K.
  //    H_v: head_size of V.
  //    D: hidden_size of Q and K, where D = N * H
  //    D_v: hidden_size of V, where D_v = N * H_v
  //    S: q_sequence_length
  //    P: past_sequence_length of kv cache
  //    L: kv_sequence_length
  //    T: total_sequence_length = P + L
  //    M: max_sequence_length of kv cache when past and present share buffer
  // ---------------------------------------------------------------
  // MultiHeadAttention inputs:
  // ---------------------------------------------------------------
  //  Q_K_V_BSNH - no packing:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D)
  //     value            (V)       : (B, L, D_v)
  //  Q_K_V_BSNH_BNSH_BNSH - cross attention (kv cache is not used, L == T, D == D_v):
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, N, L, H)
  //     value            (V)       : (B, N, L, H_v)
  //  Q_KV_BSNH_BSN2H - packed kv (kv cache is not used, bias is not allowed for packed kv):
  //     query            (Q)       : (B, S, D)
  //     key              (K/V)     : (B, L, N, 2, H)
  //     value                      : None
  //  QKV_BSN3H - packed qkv (kv cache is not used, S == L, D == D_v):
  //     query            (Q/K/V)   : (B, S, N, 3, H)
  //     key                        : None
  //     value                      : None
  //
  //  Other inputs:
  //     bias             (Q/K/V)   : None or (D + D + D_v)
  //     key_padding_mask (K/V)     : (B) or (3 * B + 2) or (B, T) or (B, S, T)
  //     attention_bias             : (B, N, S, T), (1, N, S, T), (B, 1, S, T) or (1, 1, S, T)
  //     past_key                   : (B, N, P, H) or None. Past state is only allowed for Q_K_V_BSNH.
  //     past_value                 : (B, N, P, H) or None. Past state is only allowed for Q_K_V_BSNH.
  // ---------------------------------------------------------------
  // DecoderMaskedMultiHeadAttention inputs (S == 1, D == D_v):
  // ---------------------------------------------------------------
  //  Q_K_V_BSNH - no packing:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D)
  //     value            (V)       : (B, L, D)
  //  Q_K_V_BSNH_BNSH_BNSH - cross attention (kv cache and attention_bias are not used. L == T):
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, N, L, H)
  //     value            (V)       : (B, N, L, H)
  //  QKV_BS3NH - packed qkv (S == L):
  //     query            (Q)       : (B, S, 3 * D)
  //     key              (K)       : None
  //     value            (V)       : None
  //
  //  Other inputs:
  //     bias             (Q/K/V)   : None or (3 * D)
  //     key_padding_mask (K/V)     : None or (B, T)
  //     attention_bias     : (1, N, S, T), or (B, N, S, T) where only 1 x N x S x T data is used in CUDA.
  //
  //  The following inputs are not used in cross attention (so they are None for cross attention):
  //     past_key                   : (B, N, P, H), or (B, N, M, H) when past_present_share_buffer is True.
  //                                  For CUDA, past_present_share_buffer is always True. ROCm supports both.
  //     past_value                 : (B, N, P, H), or (B, N, M, H) when past_present_share_buffer is True.
  //                                  For CUDA, past_present_share_buffer is always True. ROCm supports both.
  //     past_sequence_length       : scalar (1) when past_present_share_buffer is True.
  //  CUDA version has extra inputs (beam_width, cache_indirection) that are not checked in the class.
  //  For ROCm, see contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh for more details.
  // ---------------------------------------------------------------
  AttentionQkvFormat qkv_format = UNKNOWN;

  const auto& query_dims = query->Shape().GetDims();

  int query_rank = static_cast<int>(query_dims.size());
  if (query_rank != 3 && query_rank != 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 or 5 dimensions, got ",
                           query_rank);
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  bool dmmha_packing = operator_type == kDecoderMaskedMultiHeadAttention && key == nullptr && value == nullptr;
  int hidden_size = (query_rank == 3)
                        ? (dmmha_packing ? (static_cast<int>(query_dims[2]) / 3) : static_cast<int>(query_dims[2]))
                        : (num_heads * static_cast<int>(query_dims[4]));
  int head_size = static_cast<int>(hidden_size) / num_heads;
  int kv_sequence_length = sequence_length;

  int v_hidden_size = hidden_size;
  if (key != nullptr) {
    if (value == nullptr) {
      ORT_RETURN_IF_ERROR(Check_Q_KV<T>(query, key, num_heads, head_size, qkv_format, kv_sequence_length));
    } else {
      ORT_RETURN_IF_ERROR(Check_Q_K_V<T>(query, key, value, num_heads, head_size,
                                         qkv_format, kv_sequence_length, v_hidden_size));
    }
  } else if (value == nullptr) {  // no key and value
    ORT_RETURN_IF_ERROR(Check_QKV<T>(query, qkv_format));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value' shall absent when 'key' is absent");
  }

  int past_sequence_length = 0;
  int max_sequence_length = 0;
  if (past_key != nullptr && past_value != nullptr) {
    ORT_RETURN_IF_ERROR(CheckPast(past_key, past_value, past_seq_len,
                                  batch_size, num_heads, head_size, past_present_share_buffer,
                                  past_sequence_length, max_sequence_length));
  } else if (past_key != nullptr || past_value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' and 'past_value' shall be both present or both absent");
  }

  if (operator_type == kMultiHeadAttention) {
    if (qkv_format == AttentionQkvFormat::QKV_BS3NH) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Packed qkv of 3D BS3NH format is not support by MultiHeadAttention");
    }

    if (qkv_format == AttentionQkvFormat::Q_KV_BSNH_BSN2H && bias != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' shall be empty when packed kv is used");
    }
  }

  if (bias != nullptr) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                             bias_dims.size());
    }

    int expected_bias_length = 2 * hidden_size + v_hidden_size;
    if (bias_dims[0] != static_cast<int64_t>(expected_bias_length)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' length is expected to be 2 * hidden_size + hidden_size_v, got ",
                             bias_dims.size());
    }
  }

  int total_sequence_length = past_sequence_length + kv_sequence_length;
  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  if (key_padding_mask != nullptr) {
    mask_type = GetMaskType(key_padding_mask, batch_size, sequence_length, total_sequence_length);
    if (mask_type == AttentionMaskType::MASK_UNKNOWN) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key_padding_mask' shape is not expected.");
    }
  }

  gsl::span<const int64_t> attention_bias_dims;
  if (attention_bias != nullptr) {
    attention_bias_dims = attention_bias->Shape().GetDims();
    ORT_RETURN_IF_ERROR(CheckAttentionBias(
        attention_bias_dims, batch_size, num_heads, sequence_length, total_sequence_length));
  }

  assert(qkv_format != UNKNOWN);

  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->past_sequence_length = past_sequence_length;
    output_parameters->kv_sequence_length = kv_sequence_length;
    output_parameters->total_sequence_length = total_sequence_length;
    output_parameters->max_sequence_length = past_present_share_buffer ? max_sequence_length : total_sequence_length;
    output_parameters->input_hidden_size = 0;
    output_parameters->hidden_size = hidden_size;
    output_parameters->v_hidden_size = v_hidden_size;
    output_parameters->head_size = hidden_size / num_heads;
    output_parameters->v_head_size = v_hidden_size / num_heads;
    output_parameters->num_heads = num_heads;
    output_parameters->is_unidirectional = is_unidirectional;
    output_parameters->past_present_share_buffer = past_present_share_buffer;
    output_parameters->mask_filter_value = mask_filter_value;
    output_parameters->mask_type = mask_type;
    output_parameters->scale = scale;
    output_parameters->broadcast_attn_bias_dim_0 = attention_bias_dims.size() > 0 && attention_bias_dims[0] == 1;
    output_parameters->broadcast_attn_bias_dim_1 = attention_bias_dims.size() > 1 && attention_bias_dims[1] == 1;
    output_parameters->qkv_format = qkv_format;
  }

  return Status::OK();
}

template <typename T>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* bias,
                   const T* key_padding_mask,
                   const T* attention_bias,
                   const T* past_key,
                   const T* past_value,
                   const T* past_seq_len,
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   float scale,
                   bool is_unidirectional,
                   bool past_present_share_buffer,
                   AttentionType operator_type,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, bias, key_padding_mask, attention_bias, past_key, past_value,
                     past_seq_len, parameters, num_heads, mask_filter_value, scale, is_unidirectional,
                     past_present_share_buffer, operator_type);
}

}  // namespace multihead_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
