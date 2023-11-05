// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace group_query_attention_helper {

Status CheckInputs(const Tensor* query,
                   const Tensor* key,
                   const Tensor* value,
                   const Tensor* past_key,
                   const Tensor* past_value,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const Tensor* attention_mask,
                   bool is_past_bsnh,
                   float scale) {
  // Note: Here S* is past_cache_sequence_length, S- is past_sequence_length, S+ is sequence_length
  //     past_key                   : (B, N_k, S*, H) or (B, N_k, S-, H)
  //     past_value                 : (B, N_k, S*, H) or (B, N_k, S-, H)
  // no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, S, D_kv)
  //     value            (V)       : (B, S, D_kv)
  ORT_UNUSED_PARAMETER(value);

  AttentionQkvFormat qkv_format = Q_K_V_BSNH;
  AttentionQkvFormat past_kv_format = is_past_bsnh ? Q_K_V_BSNH : Q_K_V_BNSH;

  const auto& query_dims = query->Shape().GetDims();
  const auto& key_dims = key->Shape().GetDims();

  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int q_hidden_size = static_cast<int>(query_dims[2]);
  int head_size = static_cast<int>(q_hidden_size) / num_heads;

  int kv_hidden_size = static_cast<int>(key_dims[2]);

  int32_t past_sequence_length = 0;
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

    // BNSH
    if (!is_past_bsnh) {
      if (past_key_dims[2] != past_value_dims[2]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "BNSH Input 'past_key' and 'past_value' should have same dimension 2 (max sequence"
                               "length or past sequence length), got ",
                               past_key_dims[1]);
      }
      if (past_key_dims[1] != kv_num_heads) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_key' shall have kv_num_heads");
      }
      if (past_value_dims[1] != kv_num_heads) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_value' shall have kv_num_heads");
      }
      // We assume all sequence in past kv are right-padded to max or past sequence length
      past_sequence_length = static_cast<int>(past_key_dims[2]);
      // BSNH
    } else {
      if (past_key_dims[1] != past_value_dims[1]) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "BNSH Input 'past_key' and 'past_value' should have same dimension 1 (max sequence"
                               "length or past sequence length), got ",
                               past_key_dims[1]);
      }
      if (past_key_dims[2] != kv_num_heads) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_key' shall have kv_num_heads");
      }
      if (past_value_dims[2] != kv_num_heads) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Input 'past_value' shall have kv_num_heads");
      }
      // We assume all sequence in past kv are right-padded to max or past sequence length
      past_sequence_length = static_cast<int>(past_key_dims[1]);
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
  } else if (past_key != nullptr || past_value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' and 'past_value' shall be both present or both absent.");
  }
  // } else if (kv_share_buffer && past_key == nullptr && past_value == nullptr) {
  //   return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
  //                          "Input 'past_key' and 'past_value' shall be present when kv_share_buffer is on.");
  // }

  if (key_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_dims.size());
  }
  if (query_dims[0] != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (batch size)");
  }

  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  const auto& value_dims = value->Shape().GetDims();
  if (value_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                           value_dims.size());
  }

  if (query_dims[0] != value_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'value' shall have same dim 0 (batch_size)");
  }

  if (static_cast<int64_t>(sequence_length) != value_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query,' 'key,' and 'value' shall have the same dim 1 (sequence_length)");
  }

  if (value_dims[2] != kv_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have same hidden size as key.");
  }

  // Surmise total sequence lengths and is_prompt from the attention_mask.
  bool is_prompt = false;
  const auto& attention_mask_shape = attention_mask->Shape().GetDims();
  if (attention_mask_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "attention_mask dim 0 must be batch_size.");
  }
  if (attention_mask_shape[1] == sequence_length) {
    is_prompt = true;
  }
  int mask_sequence_length = int(attention_mask_shape[1]);
  int present_sequence_length = std::max(mask_sequence_length, past_sequence_length);

  if (parameters != nullptr) {
    GroupQueryAttentionParameters* output_parameters = reinterpret_cast<GroupQueryAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;                  // sequence length of Q
    output_parameters->seqlen_past_kv_cache = past_sequence_length;        // max sequence length of past kv tensors
    output_parameters->seqlen_present_kv_cache = present_sequence_length;  // max sequence length of present kv tensors
    output_parameters->mask_sequence_length = mask_sequence_length;
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->head_size = q_hidden_size / num_heads;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->is_unidirectional = true;
    output_parameters->is_prompt = is_prompt;
    output_parameters->scale = scale;
    output_parameters->qkv_format = qkv_format;
    output_parameters->past_kv_format = past_kv_format;
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
                   const T* attention_mask,
                   bool is_past_bsnh,
                   float scale,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, past_key, past_value, parameters, num_heads, kv_num_heads, attention_mask, is_past_bsnh, scale);
}

}  // namespace group_query_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
