// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime::contrib::paged_attention {

Status CheckInputs(const Tensor* query,
                   const Tensor* key,
                   const Tensor* value,
                   const Tensor* key_cache,
                   const Tensor* value_cache,
                   const Tensor* block_tables,
                   const Tensor* slot_mappings,
                   const Tensor* cos_cache,
                   const Tensor* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const Tensor* seqlens_k,
                   const Tensor* total_seqlen,
                   float scale) {
  // Shapes:
  //    Packed QKV:
  //        query (packed) : [batch_size, sequence_length, hidden_size + 2 * kv_hidden_size]
  //        key (packed)   : nullptr
  //        value (packed) : nullptr
  //    Non-packed QKV:
  //        query (non packed) : [batch_size, sequence_length, hidden_size]
  //        key (non packed)   : [batch_size, sequence_length, kv_hidden_size]
  //        value (non packed) : [batch_size, sequence_length, kv_hidden_size]
  //    Key/Value Cache:
  //        key_cache   : [num_blocks, block_size * num_kv_heads * head_size]
  //        value_cache : [num_blocks, block_size * num_kv_heads * head_size]
  //    Block Tables:
  //        block_tables : [batch_size, max_num_blocks_per_sequence]
  //        Assume the block tables provided are:
  //        [
  //            [0, 1, 2, -1],
  //            [3, 7, 9, -1],
  //            [4, 5, 6, 8]
  //        ]
  //        This implies that the sequence at index 0 has its kv cache stored in blocks with ids [0, 1, 2],
  //        the sequence at index 1 has its kv cache stored in blocks with ids [3, 7, 9], and and
  //        the sequence at index 2 has its kv cache stored in blocks with ids [4, 5, 6, 8].
  //        Where -1 is used to pad the block tables to the max blocks per sequence from the given sequences.
  //    Slot Mappings:
  //        slot_mappings : [num_tokens]
  //        During prompt stage num_tokens = sum(seqlens_k)
  //        During token generation stage num_tokens = batch_size
  //        Assume that the block size is 16 and that the slot tables provided are
  //        [35, 36, 2, 3, 17, 18, 19]
  //        and seqlens_k is [2, 2, 3] to indicate that there are 3 sequences with input tokens
  //        of length 2, 2, and 3 respectively.
  //        This implies that we have to write a total of 7 tokens to the kv cache
  //        at slot indices 2 (35 % 16 -1) and 3 (36 % 16 -1) of block 2 (35 / 16) for the first sequence,
  //        at slot indices 1 (2 %1 16 -1) and 2 (3 % 16 -1) of block 0 (2 / 16) for the second sequence,
  //        and at slot indices 0 (17 % 16 -1), 1 (18 % 16 -1), and 2 (19 % 16 -1) of block 1 (17 / 16)
  //        for the third sequence.
  //    Cosine and Sine Cache: (todo)
  //        cos_cache : [max_sequence_length, num_heads * head_size / 16 * 8]
  //        sin_cache : [max_sequence_length, num_heads * head_size / 16 * 8]
  //    TODO (others)

  ORT_UNUSED_PARAMETER(value);
  ORT_UNUSED_PARAMETER(block_tables);
  ORT_UNUSED_PARAMETER(slot_mappings);

  const bool is_packed_qkv = key == nullptr;
  const auto& query_dims = query->Shape().GetDims();

  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int q_hidden_size = static_cast<int>(query_dims[2]);
  int head_size = 0;

  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  int kv_hidden_size = 0;
  // Check key and value when not packed
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
  } else {
    // Check packed qkv
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

  // Check KV cache
  int32_t past_sequence_length = 0;
  if (key_cache == nullptr || value_cache == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input key_cache and value_cache are expected to be provided.");
  }

  const auto& key_cache_dims = key_cache->Shape().GetDims();
  const auto& value_cache_dims = value_cache->Shape().GetDims();

  if (key_cache_dims.size() != 24) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' is expected to have 2 dimensions, got ",
                           key_cache_dims.size());
  }
  if (value_cache_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' is expected to have 2 dimensions, got ",
                           value_cache_dims.size());
  }

  if (key_cache_dims[1] != 16 * kv_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' dimension 1 should be same as block_size * kv_hidden_size * , got ",
                           key_cache_dims[1]);
  }
  if (value_cache_dims[1] != 16 * kv_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 3 should be same as block_size * kv_hidden_size, got ",
                           value_cache_dims[1]);
  }

  // Check seqlens_k tensor (holding past seqlen for token gen)
  const auto& seqlens_dim = seqlens_k->Shape().GetDims();
  if (seqlens_dim.size() != 1 && seqlens_dim[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "seqlens_k must be shape (batch_size).");
  }

  // Set present sequence length and kv_share_buffer from input total_seqlen tensor
  if (!onnxruntime::IsScalarOr1ElementVector(total_seqlen)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "total_sequence_length tensor must be of one element.");
  }
  int total_sequence_length = *((*total_seqlen).template Data<int32_t>());
  int present_sequence_length = std::max(total_sequence_length, past_sequence_length);

  int rotary_dim = 0;
  if (cos_cache != nullptr && sin_cache != nullptr) {
    const auto& cos_dims = cos_cache->Shape().GetDims();
    const auto& sin_dims = sin_cache->Shape().GetDims();

    if (head_size % 16 != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "head_size shall be a multiple of 16. Got head_size % 16 == ",
                             head_size % 16);
    }
    if (cos_dims[0] < present_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "cos_cache dimension 0 should be of max_sequence_length.");
    }
    if (sin_dims[0] < present_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "sin_cache dimension 0 should be of max_sequence_length.");
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

  bool is_prompt = sequence_length != 1;

  if (parameters != nullptr) {
    GroupQueryAttentionParameters* output_parameters = reinterpret_cast<GroupQueryAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;                  // sequence length of Q
    output_parameters->seqlen_past_kv_cache = past_sequence_length;        // max sequence length of past kv tensors
    output_parameters->seqlen_present_kv_cache = present_sequence_length;  // max sequence length of present kv tensors
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->head_size = head_size;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->rotary_dim = rotary_dim;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->is_unidirectional = true;
    output_parameters->is_prompt = is_prompt;
    output_parameters->scale = scale;
  }

  return Status::OK();
}

Status CheckInputs(const Tensor* query,
                   const Tensor* key,
                   const Tensor* value,
                   const Tensor* key_cache,
                   const Tensor* value_cache,
                   const Tensor* block_tables,
                   const Tensor* slot_mappings,
                   const Tensor* cos_cache,
                   const Tensor* sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const Tensor* seqlens_k,
                   const Tensor* total_seqlen,
                   float scale,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ",
                           max_threads_per_block);
  }

  return CheckInputs(query, key, value, key_cache, value_cache,
                     block_tables, slot_mappings, cos_cache,
                     sin_cache, parameters, num_heads, kv_num_heads,
                     seqlens_k, total_seqlen, scale);
}

}  // namespace onnxruntime::contrib::paged_attention
