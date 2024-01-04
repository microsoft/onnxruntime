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
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* bias,
                   const T* key_padding_mask,
                   const T* relative_position_bias,
                   const T* past_key,
                   const T* past_value,
                   const T* past_seq_len,
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   float scale,
                   bool past_present_share_buffer,
                   bool dmmha_packing) {
  //     key_padding_mask (K/V)     : (B) or (2*B + 1) or (B, L) or None
  //     relative_position_bias     : (B, 1, S, L)
  //     past_key                   : (B, N, S*, H)
  //     past_value                 : (B, N, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D) or (B, N, S*, H)
  //     value            (V)       : (B, L, D_v) or (B, N, S*, H)
  //     bias             (Q/K/V)   : (D + D + D_v)
  // When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None
  // When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None or (D + D + D_v)

  AttentionQkvFormat qkv_format;

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 3 && query_dims.size() != 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 or 5 dimensions, got ",
                           query_dims.size());
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int hidden_size = (query_dims.size() == 3)
                        ? (dmmha_packing ? (static_cast<int>(query_dims[2]) / 3) : static_cast<int>(query_dims[2]))
                        : (num_heads * static_cast<int>(query_dims[4]));
  int head_size = static_cast<int>(hidden_size) / num_heads;
  int kv_sequence_length = sequence_length;

  int past_sequence_length = 0;
  int max_sequence_length = 0;
  if (past_key != nullptr && past_value != nullptr) {
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
    max_sequence_length = static_cast<int>(past_key_dims[2]);
    if (past_present_share_buffer) {
      if (past_seq_len == nullptr || !onnxruntime::IsScalarOr1ElementVector(past_seq_len)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "past_sequence_length tensor must be of one element when past_present_share_buffer is set");
      }
      past_sequence_length = *((*past_seq_len).template Data<int32_t>());
    }
  } else if (past_key != nullptr || past_value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' and 'past_value' shall be both present or both absent");
  }

  if (key != nullptr) {
    if (query_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions when key is given, got ",
                             query_dims.size());
    }

    const auto& key_dims = key->Shape().GetDims();
    if (key_dims.size() != 3 && key_dims.size() != 4 && key_dims.size() != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3, 4, or 5 dimensions, got ",
                             key_dims.size());
    }
    if (query_dims[0] != key_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'key' shall have same dim 0 (batch size)");
    }

    if (key_dims.size() == 3) {
      if (key_dims[2] != query_dims[2]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'query' and 'key' shall have same dim 2 (hidden_size)");
      }

      qkv_format = Q_K_V_BSNH;
      kv_sequence_length = static_cast<int>(key_dims[1]);
    } else if (key_dims.size() == 5) {
      if (static_cast<int>(key_dims[2]) != num_heads || static_cast<int>(key_dims[3]) != 2 || static_cast<int>(key_dims[4]) != head_size) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Expect 'key' shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv");
      }
      if (value != nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expect 'value' be none when 'key' has packed kv format.");
      }

      qkv_format = Q_KV_BSNH_BSN2H;
      kv_sequence_length = static_cast<int>(key_dims[1]);
    } else {  // key_dims.size() == 4 (cross-attention with past_key)
      if (static_cast<int>(key_dims[1]) != num_heads || static_cast<int>(key_dims[3]) != head_size) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Expect 'key' shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key");
      }

      qkv_format = UNKNOWN;
      kv_sequence_length = static_cast<int>(key_dims[2]);
    }
  } else {  // packed QKV
    if (query_dims.size() != 3 && query_dims.size() != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 or 5 dimensions when key is empty, got ",
                             query_dims.size());
    }
    if (query_dims.size() == 5 && (static_cast<int>(query_dims[2]) != num_heads || static_cast<int>(query_dims[3]) != 3)) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Expect 'query' shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv");
    }

    qkv_format = QKV_BSN3H;
  }

  if (bias != nullptr) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                             bias_dims.size());
    }

    if (value == nullptr) {
      // Currently, bias is not allowed for packed KV. This constraint can be removed later.
      // Here we assume that fusion tool will not include bias for packed KV.
      if (query_dims.size() == 5 && query_dims[3] == 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "'bias' is not allowed for packed kv. ");
      }
    }
  }

  int total_sequence_length = past_sequence_length + kv_sequence_length;
  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  if (key_padding_mask != nullptr) {
    mask_type = AttentionMaskType::MASK_UNKNOWN;
    const auto& mask_dims = key_padding_mask->Shape().GetDims();
    if (mask_dims.size() == 1) {
      if (mask_dims[0] == static_cast<int64_t>(batch_size)) {
        mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
      } else if (mask_dims[0] == static_cast<int64_t>(3) * static_cast<int64_t>(batch_size) + static_cast<int64_t>(2)) {
        mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START;
      }
    } else if (mask_dims.size() == 2 && mask_dims[0] == static_cast<int64_t>(batch_size) &&
               mask_dims[1] == static_cast<int64_t>(kv_sequence_length)) {
      mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
    } else if (mask_dims.size() == 2 && mask_dims[0] == static_cast<int64_t>(batch_size) &&
               mask_dims[1] == static_cast<int64_t>(total_sequence_length)) {
      mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
    } else if (mask_dims.size() == 3 && mask_dims[0] == static_cast<int64_t>(batch_size) &&
               mask_dims[1] == static_cast<int64_t>(sequence_length) &&
               mask_dims[2] == static_cast<int64_t>(total_sequence_length)) {
      mask_type = AttentionMaskType::MASK_3D_ATTENTION;
    }

    if (mask_type == AttentionMaskType::MASK_UNKNOWN) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key_padding_mask' shape shall be 1D, 2D, or 3D");
    }
  }

  // NOTE: In Cross-Attention, we pass the past key and value to 'key' and 'value' instead of 'past_key' and 'past_value'.
  bool pass_past_in_kv = false;
  int v_hidden_size = hidden_size;
  if (value != nullptr) {
    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 3 && value_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 or 4 dimensions, got ",
                             value_dims.size());
    }

    if (query_dims[0] != value_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'value' shall have same dim 0 (batch_size)");
    }

    if (value_dims.size() == 3) {
      if (static_cast<int64_t>(kv_sequence_length) != value_dims[1]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'key' and 'value' shall have the same dim 1 (kv_sequence_length)");
      }
      v_hidden_size = static_cast<int>(value_dims[2]);
    } else {  // value_dims.size() == 4
      if (static_cast<int64_t>(kv_sequence_length) != value_dims[2]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_key' and 'past_value' shall have the same dim 2 (kv_sequence_length)");
      }
      v_hidden_size = static_cast<int>(value_dims[1]) * static_cast<int>(value_dims[3]);
      pass_past_in_kv = true;
    }
  }

  bool broadcast_res_pos_bias = false;
  if (relative_position_bias != nullptr) {
    const auto& relative_position_bias_dims = relative_position_bias->Shape().GetDims();

    if (relative_position_bias_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' is expected to have 4 dimensions, got ",
                             relative_position_bias_dims.size());
    }
    if (relative_position_bias_dims[0] != batch_size && relative_position_bias_dims[0] != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 0 should be batch_size or 1, got ",
                             relative_position_bias_dims[0]);
    }
    if (relative_position_bias_dims[0] == 1) {
      broadcast_res_pos_bias = true;
    }
    if (relative_position_bias_dims[1] != num_heads) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 1 should be same as number of heads, got ",
                             relative_position_bias_dims[1]);
    }
    if (relative_position_bias_dims[2] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 2 should be same as sequence_length, got ",
                             relative_position_bias_dims[2]);
    }
    if (relative_position_bias_dims[3] != total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 3 should be same as total_sequence_length, got ",
                             relative_position_bias_dims[3]);
    }
  }

  // TODO: ORT_RETURN_IF(qkv_format == UNKNOWN, "Unrecognized QKV format");
  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->past_sequence_length = past_sequence_length;
    output_parameters->kv_sequence_length = kv_sequence_length;
    output_parameters->total_sequence_length = total_sequence_length;
    output_parameters->max_sequence_length = max_sequence_length;
    output_parameters->input_hidden_size = 0;
    output_parameters->hidden_size = hidden_size;
    output_parameters->v_hidden_size = v_hidden_size;
    output_parameters->head_size = hidden_size / num_heads;
    output_parameters->v_head_size = v_hidden_size / num_heads;
    output_parameters->num_heads = num_heads;
    output_parameters->is_unidirectional = false;
    output_parameters->past_present_share_buffer = past_present_share_buffer;
    output_parameters->mask_filter_value = mask_filter_value;
    output_parameters->mask_type = mask_type;
    output_parameters->scale = scale;
    output_parameters->broadcast_res_pos_bias = broadcast_res_pos_bias;
    output_parameters->pass_past_in_kv = pass_past_in_kv;
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
                   const T* relative_position_bias,
                   const T* past_key,
                   const T* past_value,
                   const T* past_seq_len,
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   float scale,
                   bool past_present_share_buffer,
                   bool dmmha_packing,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, bias, key_padding_mask, relative_position_bias, past_key, past_value,
                     past_seq_len, parameters, num_heads, mask_filter_value, scale, past_present_share_buffer,
                     dmmha_packing);
}

}  // namespace multihead_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
