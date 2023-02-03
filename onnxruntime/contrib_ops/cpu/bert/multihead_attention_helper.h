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
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   int max_threads_per_block) {
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D)
  //     value            (V)       : (B, L, D_v)
  //     bias             (Q/K/V)   : (D + D + D_v)
  //     key_padding_mask (K/V)     : (B) or (B, L) or None
  //     relative_position_bias     : (B, 1, S, L)
  // When packed kv is used:
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }

  const auto& key_dims = key->Shape().GetDims();
  if (key_dims.size() != 3 && key_dims.size() != 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 or 5 dimensions, got ",
                           key_dims.size());
  }
  if (query_dims[0] != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (batch size)");
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int hidden_size = static_cast<int>(query_dims[2]);
  int head_size = static_cast<int>(hidden_size) / num_heads;
  int kv_sequence_length = static_cast<int>(key_dims[1]);

  if (key_dims.size() == 3) {
    if (key_dims[2] != query_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'key' shall have same dim 2 (hidden_size)");
    }
  } else  // if (key_dims.size() == 5)
  {
    if (static_cast<int>(key_dims[2]) != num_heads || static_cast<int>(key_dims[3]) != 2 || static_cast<int>(key_dims[4]) != head_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Expect 'key' shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv");
    }
    if (value != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expect 'value' be none when 'key' has packed kv format.");
    }
  }

  if (bias != nullptr) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                             bias_dims.size());
    }

    // Currently, bias is not allowed for packed KV. This constraint can be removed later.
    // Here we assume that fusion tool will not include bias for packed KV.
    if (value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "'bias' is not allowed for packed kv. ");
    }
  }

  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  if (key_padding_mask != nullptr) {
    mask_type = AttentionMaskType::MASK_UNKNOWN;
    const auto& mask_dims = key_padding_mask->Shape().GetDims();
    if (mask_dims.size() == 1 && mask_dims[0] == key_dims[0]) {
      mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;
    } else if (mask_dims.size() == 2 && mask_dims[0] == key_dims[0] && mask_dims[1] == key_dims[1]) {
      mask_type = AttentionMaskType::MASK_2D_KEY_PADDING;
    }

    if (mask_type == AttentionMaskType::MASK_UNKNOWN) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key_padding_mask' shape shall be (batch_size) or (batch_size, kv_sequence_length)");
    }
  }

  int v_hidden_size = hidden_size;
  if (value != nullptr) {
    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                             value_dims.size());
    }

    if (query_dims[0] != value_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'value' shall have same dim 0 (batch_size)");
    }

    if (key_dims[1] != value_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' and 'value' shall have same same dim 1 (kv_sequence_length)");
    }
    v_hidden_size = static_cast<int>(value_dims[2]);
  }

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
    if (relative_position_bias_dims[3] != kv_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'relative_position_bias' dimension 3 should be same as total_sequence_length, got ",
                             relative_position_bias_dims[3]);
    }
  }

  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->past_sequence_length = 0;
    output_parameters->kv_sequence_length = kv_sequence_length;
    output_parameters->total_sequence_length = kv_sequence_length;
    output_parameters->max_sequence_length = 0;
    output_parameters->input_hidden_size = 0;
    output_parameters->hidden_size = hidden_size;
    output_parameters->v_hidden_size = v_hidden_size;
    output_parameters->head_size = hidden_size / num_heads;
    output_parameters->v_head_size = v_hidden_size / num_heads;
    output_parameters->num_heads = num_heads;
    output_parameters->is_unidirectional = false;
    output_parameters->past_present_share_buffer = false;
    output_parameters->mask_filter_value = mask_filter_value;
    output_parameters->mask_type = mask_type;
    output_parameters->scale = 0.0f;
  }

  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return Status::OK();
}

}  // namespace multihead_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
