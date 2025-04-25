// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace paged_attention_helper {

template <typename T = Tensor>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* key_cache,
                   const T* value_cache,
                   const T* cumulative_sequence_length,
                   const T* seqlens,
                   const T* max_query_len,
                   const T* max_seq_len,
                   const T* block_table,
                   const T* slot_mappings,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   float scale,
                   float softcap) {
  // Tensor dimensions outlined in bert_defs.cc

  const bool is_packed_qkv = key == nullptr;
  AttentionQkvFormat qkv_format = is_packed_qkv ? QKV_T3NH : Q_K_V_TNH; // TODO(aciddelgado): Is this really our preferred packed format

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 2 dimensions, got ",
                           query_dims.size());
  }

  int token_count = static_cast<int>(query_dims[0]);
  int q_hidden_size = static_cast<int>(query_dims[1]);
  int head_size = 0;

  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  // Check key and value
  int kv_hidden_size = 0;
  if (!is_packed_qkv) {
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
    if (key_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 2 dimensions, got ",
                             key_dims.size());
    } else if (token_count != key_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'key' shall have same dim 0 (token count)");
    }
    kv_hidden_size = static_cast<int>(key_dims[1]);
    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 2 dimensions, got ",
                             value_dims.size());
    } else if (token_count != value_dims[0]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'query' and 'value' shall have same dim 0 (token count)");
    } else if (value_dims[1] != kv_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have same hidden size as key.");
    }
  } else {
    head_size = static_cast<int>(q_hidden_size) / (num_heads + 2 * kv_num_heads);
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
  }

  // Check KV-Cache
  int num_blocks = 0;
  int block_size = 0;
  const auto& key_cache_dims = key_cache->Shape().GetDims();
  const auto& value_cache_dims = value_cache->Shape().GetDims();
  if (key_cache_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Input 'key_cache' is expected to have 4 dimensions, got ",
                            key_cache_dims.size());
  }
  if (value_cache_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Input 'value_cache' is expected to have 4 dimensions, got ",
                            value_cache_dims.size());
  }

  num_blocks = static_cast<int>(key_cache_dims[0]);
  block_size = static_cast<int>(key_cache_dims[1]);
  // TODO(aciddelgado): block size multiple of 16
  if (block_size % 256 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_size must be a multiple of 256. Got block_size % 256 == ",
                           block_size % 256);
  }
  if (value_cache_dims[0] != num_blocks) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 0 should be num_blocks, got ",
                           value_cache_dims[0]);
  } else if (value_cache_dims[1] != block_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 1 should be block_size, got ",
                           value_cache_dims[0]);
  }

  if (key_cache_dims[2] != value_cache_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' and 'value_cache' dimension 2 (kv num heads) should be the same, got ",
                           key_cache_dims[2], " and ", value_cache_dims[2]);
  }
  if (key_cache_dims[2] != kv_num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' shall have kv_num_heads, got ",
                           key_cache_dims[2]);
  }
  if (value_cache_dims[2] != kv_num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Input 'value_cache' shall have kv_num_heads, got ",
                            value_cache_dims[2]);
  }

  if (key_cache_dims[3] != value_cache_dims[3]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' and 'value_cache' dimension 3 (head size) should be the same, got ",
                           key_cache_dims[3], " and ", value_cache_dims[3]);
  }
  if (key_cache_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' dimension 3 should be same as head_size, got ",
                           key_cache_dims[3]);
  }
  if (value_cache_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'past_value' dimension 3 should be same as head_size, got ",
                           value_cache_dims[3]);
  }

  // Check sequence length tensors
  const auto& cumulative_seqlen_dim = cumulative_sequence_length->Shape().GetDims();
  if (cumulative_seqlen_dim.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cumulative_sequence_length must be shape (batch_size + 1).");
  }
  int batch_size = static_cast<int>(cumulative_seqlen_dim[0]) - 1;

  const auto& seqlens_dim = seqlens->Shape().GetDims();
  if (seqlens_dim.size() != 1 && seqlens_dim[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "seqlens must be shape (batch_size).");
  }

  const auto& max_query_len_dim = max_query_len->Shape().GetDims();
  if (max_query_len_dim.size() != 1 || max_query_len_dim[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "max_query_len must be shape (1).");
  }
  int max_query_length = *((*max_query_len).template Data<int32_t>());

  const auto& max_seq_len_dim = max_seq_len->Shape().GetDims();
  if (max_seq_len_dim.size() != 1 || max_seq_len_dim[0] != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "max_seq_len must be shape (1).");
  }
  int max_total_sequence_length = *((*max_seq_len).template Data<int32_t>());

  // Check block table and slot mappings
  const auto& block_table_dims = block_table->Shape().GetDims();
  if (block_table_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_table must be 2D.");
  } else if (block_table_dims[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_table dimension 0 should be num_blocks, got ",
                           block_table_dims[0]);
  }
  int max_num_blocks_per_seq = static_cast<int>(block_table_dims[1]);

  const auto& slot_mappings_dim = slot_mappings->Shape().GetDims();
  if (slot_mappings_dim.size() != 1 || slot_mappings_dim[0] != token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "slot_mappings must be shape (token count), got ",
                           slot_mappings_dim[0]);
  }

  // Check rotary cache
  int rotary_dim = 0;
  if (cos_cache != nullptr && sin_cache != nullptr) {
    const auto& cos_dims = cos_cache->Shape().GetDims();
    const auto& sin_dims = sin_cache->Shape().GetDims();

    if (head_size % 16 != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "head_size shall be a multiple of 16. Got head_size % 16 == ",
                             head_size % 16);
    }
    if (cos_dims[0] < max_total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "cos_cache dimension 0 shall not be less than the maximum total sequence length.");
    }
    if (sin_dims[0] < max_total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "sin_cache dimension 0 shall not be less than the maximum total sequence length.");
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
  } else if (cos_cache != nullptr || sin_cache != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'cos_cache' and 'sin_cache' shall be both present or both absent.");
  }

  if (parameters != nullptr) {
    PagedAttentionParameters* output_parameters = reinterpret_cast<PagedAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->token_count = token_count;
    output_parameters->sequence_length = max_query_length;                  // maximum sequence length of query tensor
    output_parameters->total_sequence_length = max_total_sequence_length;   // maximum total sequence length in kv cache after new kv are appended
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->head_size = head_size;
    output_parameters->block_size = block_size;
    output_parameters->max_num_blocks_per_seq = max_num_blocks_per_seq;
    output_parameters->num_blocks = num_blocks;
    output_parameters->rotary_dim = rotary_dim;
    output_parameters->qkv_format = qkv_format;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->scale = scale;
    output_parameters->softcap = softcap;
  }

  return Status::OK();
}

template <typename T = Tensor>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* key_cache,
                   const T* value_cache,
                   const T* cumulative_sequence_length,
                   const T* seqlens,
                   const T* max_query_len,
                   const T* max_seq_len,
                   const T* block_table,
                   const T* slot_mappings,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   float scale,
                   float softcap,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query, key, value, key_cache, value_cache, cumulative_sequence_length, seqlens, max_query_len, max_seq_len, block_table, slot_mappings, cos_cache, sin_cache, parameters, num_heads, kv_num_heads, scale, softcap);
}

}  // namespace group_query_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
