// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"

namespace onnxruntime {
namespace contrib {
namespace paged_attention_helper {

template <typename T = Tensor>
Status Check_Q_K_V(const T* query, const T* key, const T* value, const int num_heads, const int kv_num_heads,
                   int& token_count, int& q_hidden_size, int& kv_hidden_size, int& head_size) {
  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 2 dimensions, got ",
                           query_dims.size());
  }
  token_count = static_cast<int>(query_dims[0]);
  q_hidden_size = static_cast<int>(query_dims[1]);
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
  if (value_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 2 dimensions, got ",
                           value_dims.size());
  } else if (token_count != value_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'value' shall have same dim 0 (token count)");
  } else if (value_dims[1] != kv_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have same hidden size as key.");
  }
  return Status::OK();
}

template <typename T = Tensor>
Status Check_QKV(const T* packed_qkv, const T* value, const int num_heads, const int kv_num_heads, int& token_count,
                 int& q_hidden_size, int& kv_hidden_size, int& head_size) {
  const auto& packed_dims = packed_qkv->Shape().GetDims();
  if (packed_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 2 dimensions, got ",
                           packed_dims.size());
  }
  token_count = static_cast<int>(packed_dims[0]);
  head_size = static_cast<int>(static_cast<int>(packed_dims[1])) / (num_heads + 2 * kv_num_heads);
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

template <typename T = Tensor>
Status CheckKVCache(const T* key_cache, const T* value_cache, const int kv_num_heads, const int head_size,
                    int& num_blocks, int& block_size) {
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
  // TODO(aciddelgado): block size multiple of 8
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
  return Status::OK();
}

template <typename T = Tensor>
Status CheckSequenceLengthTensors(const T* cumulative_sequence_length, const T* seqlens, int& batch_size) {
  const auto& cumulative_seqlen_dim = cumulative_sequence_length->Shape().GetDims();
  if (cumulative_seqlen_dim.size() != 1 || cumulative_seqlen_dim[0] < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cumulative_sequence_length must be shape (batch_size + 1).");
  }
  batch_size = static_cast<int>(cumulative_seqlen_dim[0]) - 1;

  const auto& seqlens_dim = seqlens->Shape().GetDims();
  if (seqlens_dim.size() != 1 && seqlens_dim[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "seqlens must be shape (batch_size).");
  }
  return Status::OK();
}

template <typename T = Tensor>
Status CheckBlockTable(const T* block_table, const int batch_size, int& max_num_blocks_per_seq) {
  const auto& block_table_dims = block_table->Shape().GetDims();
  if (block_table_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_table must be 2D.");
  } else if (block_table_dims[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_table dimension 0 should be batch_size, got ",
                           block_table_dims[0]);
  }
  max_num_blocks_per_seq = static_cast<int>(block_table_dims[1]);
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
                   const T* block_table,
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
  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  // Check query, key, and value
  int token_count = 0;
  int q_hidden_size = 0;
  int kv_hidden_size = 0;
  int head_size = 0;
  const bool is_packed_qkv = key == nullptr;
  if (!is_packed_qkv) {
    ORT_RETURN_IF_ERROR(Check_Q_K_V(query, key, value, num_heads, kv_num_heads, token_count, q_hidden_size,
                                    kv_hidden_size, head_size));
  } else {
    ORT_RETURN_IF_ERROR(Check_QKV(query, value, num_heads, kv_num_heads, token_count, q_hidden_size, kv_hidden_size,
                                  head_size));
  }

  // Check KV-Cache
  int num_blocks = 0;
  int block_size = 0;
  ORT_RETURN_IF_ERROR(CheckKVCache(key_cache, value_cache, kv_num_heads, head_size, num_blocks, block_size));

  // Check sequence length tensors
  int batch_size = 0;
  ORT_RETURN_IF_ERROR(CheckSequenceLengthTensors(cumulative_sequence_length, seqlens, batch_size));

  // Check block table and slot mappings
  int max_num_blocks_per_seq = 0;
  ORT_RETURN_IF_ERROR(CheckBlockTable(block_table, batch_size, max_num_blocks_per_seq));

  // Check rotary cache
  int rotary_dim = 0;
  if (cos_cache != nullptr && sin_cache != nullptr) {
    // 0 to bypass checking rotary cache size
    ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckRotaryCaches(cos_cache, sin_cache, head_size,
                                                                        0, rotary_dim));
  } else if (cos_cache != nullptr || sin_cache != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'cos_cache' and 'sin_cache' shall be both present or both absent.");
  }

  if (parameters != nullptr) {
    PagedAttentionParameters* output_parameters = reinterpret_cast<PagedAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->token_count = token_count;
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->head_size = head_size;
    output_parameters->block_size = block_size;
    output_parameters->max_num_blocks_per_seq = max_num_blocks_per_seq;
    output_parameters->num_blocks = num_blocks;
    output_parameters->rotary_dim = rotary_dim;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->scale = scale;
    output_parameters->softcap = softcap;
  }

  return Status::OK();
}

}  // namespace paged_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
