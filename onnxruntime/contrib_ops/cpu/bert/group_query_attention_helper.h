// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace group_query_attention_helper {

template <typename T>
Status Check_Q_K_V(const T* query, const T* key, const T* value, const int num_heads, const int kv_num_heads,
                   int& batch_size, int& sequence_length, int& q_hidden_size, int& kv_hidden_size, int& head_size) {
  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }
  batch_size = static_cast<int>(query_dims[0]);
  sequence_length = static_cast<int>(query_dims[1]);
  q_hidden_size = static_cast<int>(query_dims[2]);
  head_size = static_cast<int>(q_hidden_size) / num_heads;
  if (head_size % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "head_size must be a multiple of 8. Got head_size % 8 == ",
                           head_size % 8);
  }
  if (value == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.");
  }
  const auto& key_dims = key->Shape().GetDims();
  if (key_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_dims.size());
  } else if (query_dims[0] != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (batch size)");
  } else if (query_dims[1] != key_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 1 (sequence length)");
  }
  kv_hidden_size = static_cast<int>(key_dims[2]);
  if (kv_hidden_size % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "kv_hidden_size must be a multiple of kv_num_heads. Got kv_hidden_size % kv_num_heads == ",
                           kv_hidden_size % kv_num_heads);
  } else if (kv_hidden_size / kv_num_heads != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "kv_hidden_size / kv_num_heads must be equal to head_size. Got kv_hidden_size / kv_num_heads == ",
                           kv_hidden_size / kv_num_heads);
  }
  const auto& value_dims = value->Shape().GetDims();
  if (value_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                           value_dims.size());
  } else if (query_dims[0] != value_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'value' shall have same dim 0 (batch size)");
  } else if (query_dims[1] != value_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'value' shall have same dim 1 (sequence length)");
  } else if (value_dims[2] != kv_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have same hidden size as key.");
  }
  return Status::OK();
}

template <typename T>
Status Check_QKV(const T* packed_qkv, const T* value, const int num_heads, const int kv_num_heads, int& batch_size,
                 int& sequence_length, int& q_hidden_size, int& kv_hidden_size, int& head_size) {
  const auto& packed_dims = packed_qkv->Shape().GetDims();
  if (packed_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           packed_dims.size());
  }
  batch_size = static_cast<int>(packed_dims[0]);
  sequence_length = static_cast<int>(packed_dims[1]);
  head_size = static_cast<int>(static_cast<int>(packed_dims[2])) / (num_heads + 2 * kv_num_heads);
  // Check packed qkv
  if (head_size % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "head_size must be a multiple of 8. Got head_size % 8 == ",
                           head_size % 8);
  }
  if (value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.");
  }
  q_hidden_size = head_size * num_heads;
  kv_hidden_size = head_size * kv_num_heads;
  return Status::OK();
}

template <typename T>
Status CheckPast(const T* past_key, const T* past_value, int batch_size, int kv_num_heads, int head_size,
                 int& past_sequence_length) {
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
  return Status::OK();
}

template <typename T>
Status CheckRotaryCaches(const T* cos_cache, const T* sin_cache, int head_size, int total_sequence_length,
                         int& rotary_dim) {
  const auto& cos_dims = cos_cache->Shape().GetDims();
  const auto& sin_dims = sin_cache->Shape().GetDims();

  if (head_size % 16 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "head_size shall be a multiple of 16. Got head_size % 16 == ",
                           head_size % 16);
  }
  if (cos_dims[0] < total_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache dimension 0 shall not be less than total_sequence_length.");
  }
  if (sin_dims[0] < total_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "sin_cache dimension 0 shall not be less than total_sequence_length.");
  }
  if (cos_dims[1] > (head_size / 16) * 8 || cos_dims[1] % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache dimension 1 must be <= head_size / 2 and a multiple of 8.");
  }
  if (sin_dims[1] > (head_size / 16) * 8 || sin_dims[1] % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "sin_cache dimension 1 must be <= head_size / 2 and a multiple of 8.");
  }
  if (cos_dims[1] != sin_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache dimension 1 must be the same.");
  }
  rotary_dim = static_cast<int>(cos_dims[1] * 2);
  return Status::OK();
}

template <typename T = Tensor>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* past_key,
                   const T* past_value,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const T* seqlens_k,
                   const T* total_seqlen,
                   float scale,
                   float softcap) {
  // Note: Here S* is seqlen_past_kv_cache, S+ is seqlen_present_kv_cache
  //     past_key                   : (B, N_k, S*, H) or (B, N_k, S+, H) or nullptr
  //     past_value                 : (B, N_k, S*, H) or (B, N_k, S+, H) or nullptr
  // no packing for q/k/v:
  //     query            (Q)       : (B, S, D) or (B, S, (D_q + 2 D_kv))
  //     key              (K)       : (B, S, D_kv) or nullptr
  //     value            (V)       : (B, S, D_kv) or nullptr

  AttentionQkvFormat qkv_format = Q_K_V_BSNH;
  AttentionQkvFormat past_kv_format = Q_K_V_BNSH;

  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  int batch_size = 0;
  int sequence_length = 0;
  int q_hidden_size = 0;
  int kv_hidden_size = 0;
  int head_size = 0;
  const bool is_packed_qkv = key == nullptr;
  if (!is_packed_qkv) {
    ORT_RETURN_IF_ERROR(Check_Q_K_V(query, key, value, num_heads, kv_num_heads, batch_size, sequence_length,
                                    q_hidden_size, kv_hidden_size, head_size));
  } else {
    qkv_format = QKV_BS3NH;
    ORT_RETURN_IF_ERROR(Check_QKV(query, value, num_heads, kv_num_heads, batch_size, sequence_length, q_hidden_size,
                                  kv_hidden_size, head_size));
  }

  // Check past-present KV
  int32_t past_sequence_length = 0;
  if (past_key != nullptr && past_value != nullptr) {
    ORT_RETURN_IF_ERROR(CheckPast(past_key, past_value, batch_size, kv_num_heads, head_size, past_sequence_length));
  } else if (past_key != nullptr || past_value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_key' and 'past_value' shall be both present or both absent.");
  }

  const auto& seqlens_k_dim = seqlens_k->Shape().GetDims();
  if (seqlens_k_dim.size() != 1 && seqlens_k_dim[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "seqlens_k must be shape (batch_size).");
  }

  // Set present sequence length from input total_seqlen tensor
  if (!onnxruntime::IsScalarOr1ElementVector(total_seqlen)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "total_sequence_length tensor must be of one element.");
  }
  int total_sequence_length = *((*total_seqlen).template Data<int32_t>());
  int present_sequence_length = std::max(total_sequence_length, past_sequence_length);

  int rotary_dim = 0;
  if (cos_cache != nullptr && sin_cache != nullptr) {
    ORT_RETURN_IF_ERROR(CheckRotaryCaches(cos_cache, sin_cache, head_size, total_sequence_length, rotary_dim));
  } else if (cos_cache != nullptr || sin_cache != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'cos_cache' and 'sin_cache' shall be both present or both absent.");
  }

  bool is_subsequent_prompt = false;
  if (sequence_length > 1 && sequence_length != total_sequence_length) {
    if (batch_size != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "batch_size must be 1 when sequence_length > 1 and past context is given.");
    }
    is_subsequent_prompt = true;
  }

  bool is_first_prompt;
  if (is_subsequent_prompt) {
    is_first_prompt = false;  // irrelevant for interactive decoding
  } else {
    // If not interactive, sequence_length is 1 for token gen and arbitrarily large for prompt
    is_first_prompt = (sequence_length == total_sequence_length);
    if (!is_first_prompt && sequence_length != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "sequence_length shall be 1 when it is not prompt.");
    }
  }

  if (parameters != nullptr) {
    GroupQueryAttentionParameters* output_parameters = reinterpret_cast<GroupQueryAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;                  // sequence length of Q
    output_parameters->seqlen_past_kv_cache = past_sequence_length;        // max sequence length of past kv tensors
    output_parameters->seqlen_present_kv_cache = present_sequence_length;  // max sequence length of present kv tensors
    output_parameters->total_sequence_length = total_sequence_length;      // total sequence length
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->head_size = head_size;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->rotary_dim = rotary_dim;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->is_unidirectional = true;
    output_parameters->is_subsequent_prompt = is_subsequent_prompt;
    output_parameters->is_first_prompt = is_first_prompt;
    output_parameters->scale = scale;
    output_parameters->softcap = softcap;
    output_parameters->qkv_format = qkv_format;
    output_parameters->past_kv_format = past_kv_format;
  }

  return Status::OK();
}

template <typename T = Tensor>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* past_key,
                   const T* past_value,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const T* seqlens_k,
                   const T* total_seqlen,
                   float scale,
                   float softcap,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, past_key, past_value, cos_cache, sin_cache, parameters, num_heads, kv_num_heads, seqlens_k, total_seqlen, scale, softcap);
}

template <typename T = Tensor>
Status CheckCustomAttentionInputs(const T* position_ids,
                                  const T* attention_bias,
                                  const T* head_sink,
                                  const GroupQueryAttentionParameters& parameters) {
  if (position_ids != nullptr) {
    const auto& pos_ids_shape = position_ids->Shape();
    if (pos_ids_shape[0] != parameters.batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "position_ids dimension 0 must be equal to the batch size, got ", pos_ids_shape[0]);
    }

    if (pos_ids_shape[1] < parameters.sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "position_ids dimension 1 must be atleast sequence length, got ", pos_ids_shape[1]);
    }
  }

  if (attention_bias != nullptr) {
    const auto& attn_bias_shape = attention_bias->Shape();
    if ((attn_bias_shape[0] != parameters.batch_size) && (attn_bias_shape[0] != 1)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias dimension 0 must be equal to the batch size or 1, got ", attn_bias_shape[0]);
    }

    if ((attn_bias_shape[1] != parameters.num_heads) && (attn_bias_shape[1] != 1)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias dimension 1 must be equal to the num heads or 1, got ", attn_bias_shape[1]);
    }

    if (attn_bias_shape[2] != parameters.sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias dimension 2 must be equal to the sequence length, got ", attn_bias_shape[2]);
    }

    if (attn_bias_shape[3] != parameters.total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_bias dimension 3 must be equal to total_sequence_length, got ", attn_bias_shape[3]);
    }
  }

  if (head_sink != nullptr) {
    if (parameters.use_smooth_softmax) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "head_sink should not be provided when use_smooth_softmax is true.");
    }

    const auto& head_sink_shape = head_sink->Shape();
    if (head_sink_shape.NumDimensions() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "head_sink must be a 1D tensor");
    }

    if (head_sink_shape[0] != parameters.num_heads) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "head_sink dimension 0 must be equal to the num heads, got ", head_sink_shape[0]);
    }
  }

  return Status::OK();
}

template <typename T = Tensor>
Status CheckOutputs(const T* output_qk, int qk_output) {
  const bool is_valid_qk_output = qk_output == static_cast<int>(QKOutputType::NO_OUTPUT) ||
                                  qk_output == static_cast<int>(QKOutputType::BEFORE_SOFTMAX) ||
                                  qk_output == static_cast<int>(QKOutputType::AFTER_SOFTMAX);
  if (!is_valid_qk_output) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "qk_output attribute received unsupported value ", qk_output);
  }

  if (qk_output != static_cast<int>(QKOutputType::NO_OUTPUT) && output_qk == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qk_output attribute was configured but output buffer was not provided");
  }

  return Status::OK();
}

inline Status CheckQKOutput(int num_outputs, int qk_output) {
  if (num_outputs > 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "output_qk optional output is not supported on");
  }

  if (qk_output != static_cast<int>(QKOutputType::NO_OUTPUT)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qk_output attribute is not supported on");
  }

  return Status::OK();
}

}  // namespace group_query_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
