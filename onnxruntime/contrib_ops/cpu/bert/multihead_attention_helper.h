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
                   void* parameters,
                   int num_heads,
                   float mask_filter_value,
                   int max_threads_per_block) {
  //   query            (Q)       : (B, S, D)
  //   key              (K)       : (B, L, D)
  //   value            (V)       : (B, L, D_v)
  //   bias             (Q/K/V)   : (D + D + D_v)
  //   key_padding_mask (K/V)     : (B, L) or (L)

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }

  const auto& key_dims = key->Shape().GetDims();
  if (key_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_dims.size());
  }

  const auto& bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
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

  if (query_dims[0] != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (batch size)");
  }

  int64_t batch_size = query_dims[0];
  int64_t sequence_length = query_dims[1];
  int64_t kv_sequence_length = key_dims[1];
  int64_t q_hidden_size = query_dims[2];
  int64_t v_hidden_size = 0;

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
                           "Input 'key' and 'value' shall have same same dim 1 (sequence_length)");
  }
  v_hidden_size = value_dims[2];

  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = static_cast<int>(batch_size);
    output_parameters->sequence_length = static_cast<int>(sequence_length);
    output_parameters->past_sequence_length = 0;
    output_parameters->kv_sequence_length = static_cast<int>(kv_sequence_length);
    output_parameters->total_sequence_length = static_cast<int>(kv_sequence_length);
    output_parameters->max_sequence_length = 0;
    output_parameters->input_hidden_size = 0;
    output_parameters->hidden_size = static_cast<int>(q_hidden_size);
    output_parameters->v_hidden_size = static_cast<int>(v_hidden_size);
    output_parameters->head_size = static_cast<int>(q_hidden_size) / num_heads;
    output_parameters->v_head_size = static_cast<int>(v_hidden_size) / num_heads;
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
