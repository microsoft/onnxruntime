// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/cross_attention_base.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

Status CrossAttentionBase::CheckInputs(const Tensor* query,
                                       const Tensor* key,
                                       const Tensor* value,
                                       const Tensor* bias,
                                       void* parameters,
                                       const int max_threads_per_block) const {
  //   query         (Q)       : (B, S, D)
  //   key           (K)       : (B, L, D)    or (B, L, D + D_v) packed
  //   value         (V)       : (B, L, D_v)  or (B, L, D + D_v) packed
  //   bias          (Q/K/V)   : (D + D + D_v)

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

  const auto& value_dims = value->Shape().GetDims();
  if (value_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                           value_dims.size());
  }

  const auto& bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  if (query_dims[0] != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (batch size)");
  }

  if (query_dims[0] != value_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'value' shall have same dim 0 (batch size)");
  }

  if (key_dims[1] != value_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key' and 'value' shall have same same dim 0 (sequence length)");
  }

  int64_t batch_size = query_dims[0];
  int64_t sequence_length = query_dims[1];
  int64_t q_hidden_size = query_dims[2];

  int64_t kv_sequence_length = value_dims[1];
  int64_t v_hidden_size = value_dims[2];

  if (key == value) {  // packed key value
    if (bias_dims[0] != query_dims[2] + value_dims[2] || query_dims[2] >= value_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query', 'value' and 'bias' dim 2 not matched for packed key/value format");
    }
    v_hidden_size = value_dims[2] - query_dims[2];
  } else {
    if (query_dims[2] != key_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' and 'key' shall have same hidden size");
    }
  }

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
    output_parameters->head_size = static_cast<int>(q_hidden_size) / num_heads_;
    output_parameters->v_head_size = static_cast<int>(v_hidden_size) / num_heads_;
    output_parameters->num_heads = num_heads_;
    output_parameters->is_unidirectional = false;
    output_parameters->past_present_share_buffer = false;
  }

  if (max_threads_per_block > 0 && num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
