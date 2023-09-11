// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace group_query_attention_helper {

// TODO(aciddelgado): double-check this
template <typename T>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* past_key,
                   const T* past_value,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   float scale) {
  //     past_key                   : (B, N_k, S*, H)
  //     past_value                 : (B, N_k, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D_kv) or (B, N_k, S*, H)
  //     value            (V)       : (B, L, D_kv) or (B, N_k, S*, H)
  // CURRENTLY UNSUPPORTED!! When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  // CURRENTLY UNSUPPORTED!! When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None

  AttentionQkvFormat qkv_format;

  const auto& query_dims = query->Shape().GetDims();
  const auto& key_dims = key->Shape().GetDims();
  const auto& value_dims = value->Shape().GetDims();

  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int q_hidden_size = static_cast<int>(query_dims[2]);
  int head_size = static_cast<int>(q_hidden_size) / num_heads;

  int kv_sequence_length = sequence_length;
  int kv_hidden_size = (key_dims.size() == 3)
                        ? static_cast<int>(key_dims[2])
                        : (kv_num_heads * static_cast<int>(key_dims[3]));

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

    if (past_key_dims[1] != kv_num_heads) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'past_key' dimension 1 should be same as number of heads, got ",
                             past_key_dims[1]);
    }
    if (past_value_dims[1] != kv_num_heads) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'past_value' dimension 1 should be same as number of heads, got ",
                             past_value_dims[1]);
    }
    if (past_key_dims[2] != past_value_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'past_key' and 'past_value' shall have same dim 2 (past_sequence_length)");
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
    if (key_dims.size() != 3 && key_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 or 4 dimensions, got ",
                             key_dims.size());
    }
    if (query_dims[0] != key_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'key' shall have same dim 0 (batch size)");
    }

    if (key_dims.size() == 3) {
      if (num_heads % kv_num_heads != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                               num_heads % kv_num_heads);
      }
      if (key_dims[2] != value_dims[2]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'key' and 'value' shall have same dim 2 (kv_hidden_size)");
      }

      qkv_format = Q_K_V_BSNH;
      kv_sequence_length = static_cast<int>(key_dims[1]);
    } else {  // key_dims.size() == 4 (cross-attention with past_key)
      if (static_cast<int>(key_dims[1]) != kv_num_heads || static_cast<int>(key_dims[3]) != head_size) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Expect 'key' shape (batch_size, kv_num_heads, kv_sequence_length, head_size) for past_key");
      }

      qkv_format = UNKNOWN;
      kv_sequence_length = static_cast<int>(key_dims[2]);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Missing key tensor.");
  }

  // TODO(aciddelgado): what is this note referring to/do I support?
  // NOTE: In Cross-Attention, we pass the past key and value to 'key' and 'value' instead of 'past_key' and 'past_value'.
  // bool pass_past_in_kv = false;
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
    } else {  // value_dims.size() == 4
      if (static_cast<int64_t>(kv_sequence_length) != value_dims[2]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_key' and 'past_value' shall have the same dim 2 (kv_sequence_length)");
      }
      // pass_past_in_kv = true;
    }

    int v_hidden_size = (value_dims.size() == 3) ? value_dims[2] : value_dims[1] * value_dims[3];
    if (v_hidden_size != kv_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have same hidden size as key.");
    }

  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Missing value tensor.");
  }

  int total_sequence_length = past_sequence_length + kv_sequence_length;
  // TODO: ORT_RETURN_IF(qkv_format == UNKNOWN, "Unrecognized QKV format");
  if (parameters != nullptr) {
    GroupQueryAttentionParameters* output_parameters = reinterpret_cast<GroupQueryAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->past_sequence_length = past_sequence_length;
    output_parameters->kv_sequence_length = kv_sequence_length;
    output_parameters->total_sequence_length = total_sequence_length;
    output_parameters->max_sequence_length = max_sequence_length;
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->head_size = q_hidden_size / num_heads;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->is_unidirectional = true;
    output_parameters->do_rotary = false; // TODO(aciddelgado): huh?
    output_parameters->scale = scale;
    output_parameters->qkv_format = qkv_format;
  }

  return Status::OK();
}

template <typename T>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* past_key,
                   const T* past_value,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   float scale,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, past_key, past_value, parameters, num_heads, kv_num_heads, scale);
}

}  // namespace group_query_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
