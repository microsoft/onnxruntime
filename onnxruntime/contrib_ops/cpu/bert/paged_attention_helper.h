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

// LATENT (absorbed MLA) mode: 'query' is the absorbed query, 'key' is the latent row
// [compressed_kv; k_pe] shared by all heads, and 'value' is absent because V is the leading
// v_head_size channels of the same latent row. See docs/contrib_ops/cuda/paged_attention.md §12.
template <typename T = Tensor>
Status Check_Q_K_Latent(const T* query, const T* key, const T* value, const int num_heads, const int kv_num_heads,
                        int& token_count, int& q_hidden_size, int& kv_hidden_size, int& head_size) {
  if (key == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key' is required when 'kv_cache_layout' is 'LATENT'.");
  }
  if (value != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value' must be absent when 'kv_cache_layout' is 'LATENT': the value of every "
                           "head is the leading 'v_head_size' channels of the latent key.");
  }

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 2 dimensions, got ",
                           query_dims.size());
  }
  token_count = static_cast<int>(query_dims[0]);
  q_hidden_size = static_cast<int>(query_dims[1]);
  if (q_hidden_size % num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' hidden size must be a multiple of num_heads. Got ", q_hidden_size,
                           " % ", num_heads, " == ", q_hidden_size % num_heads);
  }
  head_size = q_hidden_size / num_heads;
  if (head_size % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "head_size must be a multiple of 8. Got head_size % 8 == ", head_size % 8);
  }

  const auto& key_dims = key->Shape().GetDims();
  if (key_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 2 dimensions, got ",
                           key_dims.size());
  }
  if (token_count != key_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'query' and 'key' shall have same dim 0 (token count)");
  }
  kv_hidden_size = static_cast<int>(key_dims[1]);
  if (kv_hidden_size != kv_num_heads * head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key' is expected to have hidden size kv_num_heads * head_size = ",
                           kv_num_heads * head_size, " in 'LATENT' mode, got ", kv_hidden_size);
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

// `value_cache` is null only in LATENT mode, where V aliases the leading channels of `key_cache`
// and there is no second physical cache to validate.
template <typename T = Tensor>
Status CheckKVCache(const T* key_cache, const T* value_cache, const int kv_num_heads, const int head_size,
                    int& num_blocks, int& block_size) {
  const auto& key_cache_dims = key_cache->Shape().GetDims();
  if (key_cache_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' is expected to have 4 dimensions, got ",
                           key_cache_dims.size());
  }

  num_blocks = static_cast<int>(key_cache_dims[0]);
  block_size = static_cast<int>(key_cache_dims[1]);
  // The op itself only needs the block size to be a power of two >= 16 (the granularity every
  // serving framework uses). The vendored FlashAttention paged kernel has a stricter,
  // head-size-dependent requirement (a kBlockN tile must not straddle a page); that is enforced at
  // backend-selection time in paged_attention.cc, which falls back to the gather-based
  // memory-efficient path instead of rejecting the model. See docs/contrib_ops/cuda/paged_attention.md §18.
  if (block_size < 16 || (block_size & (block_size - 1)) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "block_size must be a power of two and at least 16. Got block_size == ",
                           block_size);
  }

  if (key_cache_dims[2] != kv_num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' shall have kv_num_heads, got ",
                           key_cache_dims[2]);
  }
  if (key_cache_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' dimension 3 should be same as head_size, got ",
                           key_cache_dims[3]);
  }

  if (value_cache == nullptr) {
    return Status::OK();
  }

  const auto& value_cache_dims = value_cache->Shape().GetDims();
  if (value_cache_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' is expected to have 4 dimensions, got ",
                           value_cache_dims.size());
  }
  if (value_cache_dims[0] != num_blocks) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 0 should be num_blocks, got ",
                           value_cache_dims[0]);
  } else if (value_cache_dims[1] != block_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 1 should be block_size, got ",
                           value_cache_dims[1]);
  }

  if (key_cache_dims[2] != value_cache_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'key_cache' and 'value_cache' dimension 2 (kv num heads) should be the same, got ",
                           key_cache_dims[2], " and ", value_cache_dims[2]);
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
  if (value_cache_dims[3] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' dimension 3 should be same as head_size, got ",
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
  if (seqlens_dim.size() != 1 || seqlens_dim[0] != batch_size) {
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

// slot_mapping (input 10) is the scheduler-owned write map: one flat slot index per query token
// into the cache viewed as [num_blocks * block_size, kv_num_heads, head_size], or -1 to skip the
// K/V store for that token. Element range is not validated on the host: that would require a
// device-to-host copy every step. Out-of-range values are undefined behavior, exactly as for
// block_table today.
template <typename T = Tensor>
Status CheckSlotMapping(const T* slot_mapping, const int token_count) {
  const auto& dims = slot_mapping->Shape().GetDims();
  if (dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'slot_mapping' is expected to have 1 dimension, got ", dims.size());
  }
  if (dims[0] != token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'slot_mapping' dimension 0 should be token_count (", token_count,
                           "), got ", dims[0]);
  }
  return Status::OK();
}

template <typename T = Tensor>
Status CheckHeadSink(const T* head_sink, const int num_heads) {
  const auto& dims = head_sink->Shape().GetDims();
  if (dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "head_sink must be a 1D tensor");
  }
  if (dims[0] != num_heads) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "head_sink dimension 0 must be equal to the num heads, got ", dims[0]);
  }
  return Status::OK();
}

template <typename T = Tensor>
Status CheckQKNormWeights(const T* q_norm_weight, const T* k_norm_weight, const int head_size) {
  if ((q_norm_weight != nullptr) != (k_norm_weight != nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'q_norm_weight' and 'k_norm_weight' must be provided together.");
  }
  if (q_norm_weight == nullptr) {
    return Status::OK();
  }
  const auto& q_dims = q_norm_weight->Shape().GetDims();
  const auto& k_dims = k_norm_weight->Shape().GetDims();
  if (q_dims.size() != 1 || q_dims[0] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'q_norm_weight' must be a 1D tensor of shape (head_size) = (", head_size, ").");
  }
  if (k_dims.size() != 1 || k_dims[0] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'k_norm_weight' must be a 1D tensor of shape (head_size) = (", head_size, ").");
  }
  return Status::OK();
}

// Validates one side (K or V) of the quantized paged KV cache contract. `is_quantized_cache`
// reflects the element type the kernel was instantiated for, so a mismatch between the cache dtype
// and the quant-type attribute is reported instead of silently producing garbage.
template <typename T = Tensor>
Status CheckKVCacheQuantization(const T* scale, const char* scale_name, const char* quant_type_name,
                                const KVQuantizationType quant_type, const bool is_quantized_cache,
                                const int kv_num_heads, const int head_size) {
  if (quant_type == KVQuantizationType::NONE) {
    if (scale != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input '", scale_name, "' must not be provided when '", quant_type_name, "' is 'NONE'.");
    }
    if (is_quantized_cache) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The KV cache has a quantized element type, so '", quant_type_name,
                             "' must be 'PER_TENSOR' or 'PER_CHANNEL'.");
    }
    return Status::OK();
  }

  if (!is_quantized_cache) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'", quant_type_name,
                           "' is set, but the KV cache element type is not quantized. "
                           "Use an int8 or float8e4m3fn cache, or set '",
                           quant_type_name, "' to 'NONE'.");
  }
  if (scale == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input '", scale_name, "' is required when '", quant_type_name, "' is not 'NONE'.");
  }

  const auto& dims = scale->Shape().GetDims();
  const int64_t count = scale->Shape().Size();
  if (quant_type == KVQuantizationType::PER_TENSOR) {
    if (count != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input '", scale_name, "' must have exactly 1 element for PER_TENSOR quantization, got ",
                             count);
    }
    return Status::OK();
  }

  // PER_CHANNEL. The canonical shape is (kv_num_heads, 1, head_size), matching GroupQueryAttention;
  // any shape with the same element count and a trailing head_size is accepted so that callers may
  // pass (kv_num_heads, head_size) directly.
  if (count != static_cast<int64_t>(kv_num_heads) * head_size || dims.empty() || dims.back() != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input '", scale_name, "' must have shape (kv_num_heads, 1, head_size) = (",
                           kv_num_heads, ", 1, ", head_size, ") for PER_CHANNEL quantization, got ",
                           scale->Shape().ToString());
  }
  return Status::OK();
}

// Validates one side (K or V) of the `k_cache_dtype` / `v_cache_dtype` contract against
// `storage_dtype`, the element type the kernel was instantiated for. DEFAULT means "the cache
// tensor's element type is also the logical type" and always passes; naming that same type
// explicitly is allowed but must agree. The sub-byte members describe a logical type packed two per
// byte into a uint8 cache; the schema reserves them, but no backend decodes them yet, so they are
// rejected here instead of being silently mis-read. See docs/contrib_ops/cuda/paged_attention.md §8.
inline Status CheckKVCacheDataType(const KVCacheDataType cache_dtype, const KVCacheDataType storage_dtype,
                                   const char* attr_name) {
  if (cache_dtype == KVCacheDataType::DEFAULT || cache_dtype == storage_dtype) {
    return Status::OK();
  }
  if (IsSubByteKVCacheDataType(cache_dtype)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'", attr_name, "' == '", KVCacheDataTypeToString(cache_dtype),
                           "' requires a uint8 packed cache, which is not enabled in this build.");
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "'", attr_name, "' is '", KVCacheDataTypeToString(cache_dtype),
                         "', but the cache tensor's element type is '", KVCacheDataTypeToString(storage_dtype),
                         "'. Leave the attribute at '' to use the tensor's element type.");
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
                   const T* slot_mapping,
                   const T* head_sink,
                   const T* q_norm_weight,
                   const T* k_norm_weight,
                   const T* k_scale,
                   const T* v_scale,
                   const T* attention_metadata,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   float scale,
                   float softcap,
                   float qk_norm_epsilon,
                   KVQuantizationType k_quant_type,
                   KVQuantizationType v_quant_type,
                   KVCacheDataType k_cache_dtype,
                   KVCacheDataType v_cache_dtype,
                   KVCacheDataType cache_storage_dtype,
                   bool is_latent_kv,
                   int v_head_size_attr,
                   int rotary_offset,
                   bool has_explicit_scale,
                   int max_threads_per_block) {
  const bool is_quantized_cache = IsQuantizedKVCacheDataType(cache_storage_dtype);
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }
  if (num_heads % kv_num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ",
                           num_heads % kv_num_heads);
  }

  // Check query, key, and value. `kv_cache_layout` is inspected before the presence pattern,
  // because LATENT's "key present, value absent" pattern would otherwise be indistinguishable from
  // an ill-formed SEPARATE node. See docs/contrib_ops/cuda/paged_attention.md §4.6.
  int token_count = 0;
  int q_hidden_size = 0;
  int kv_hidden_size = 0;
  int head_size = 0;
  const bool is_packed_qkv = !is_latent_kv && key == nullptr;
  if (is_latent_kv) {
    ORT_RETURN_IF_ERROR(Check_Q_K_Latent(query, key, value, num_heads, kv_num_heads, token_count, q_hidden_size,
                                         kv_hidden_size, head_size));
  } else if (!is_packed_qkv) {
    ORT_RETURN_IF_ERROR(Check_Q_K_V(query, key, value, num_heads, kv_num_heads, token_count, q_hidden_size,
                                    kv_hidden_size, head_size));
  } else {
    ORT_RETURN_IF_ERROR(Check_QKV(query, value, num_heads, kv_num_heads, token_count, q_hidden_size, kv_hidden_size,
                                  head_size));
  }

  // Effective V head size (§12.2). A V width that differs from head_size is only meaningful when V
  // is a slice of the latent key, so it is confined to LATENT mode: no SEPARATE-mode backend
  // supports asymmetric K/V widths, and value_cache's last dimension is head_size by construction.
  int v_head_size = head_size;
  if (v_head_size_attr != 0) {
    if (v_head_size_attr < 1 || v_head_size_attr > head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "'v_head_size' must be 0 (meaning head_size) or in [1, head_size] = [1, ", head_size,
                             "], got ", v_head_size_attr);
    }
    if (v_head_size_attr != head_size && !is_latent_kv) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "'v_head_size' (", v_head_size_attr, ") may only differ from head_size (", head_size,
                             ") when 'kv_cache_layout' is 'LATENT'.");
    }
    v_head_size = v_head_size_attr;
  }
  // The softmax-scale trap (§12.6): DeepSeek derives its scale from the pre-absorption head width,
  // so the 1/sqrt(head_size) default would silently produce plausible-but-wrong logits.
  if (v_head_size != head_size && !has_explicit_scale) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "An explicit 'scale' attribute is required when 'v_head_size' (", v_head_size,
                           ") differs from head_size (", head_size,
                           "): the default 1/sqrt(head_size) is not the intended scale for absorbed MLA.");
  }

  if (is_latent_kv) {
    if (value_cache != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'value_cache' must be absent when 'kv_cache_layout' is 'LATENT': the value cache "
                             "is the leading 'v_head_size' channels of 'key_cache'.");
    }
    if (kv_num_heads != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "'kv_num_heads' must be 1 when 'kv_cache_layout' is 'LATENT', got ", kv_num_heads);
    }
    // §12.9: both combinations are well defined mathematically but no MLA model uses them, and an
    // untested silent result is worse than a rejection.
    if (head_sink != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'head_sink' is not supported when 'kv_cache_layout' is 'LATENT'.");
    }
    if (q_norm_weight != nullptr || k_norm_weight != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'q_norm_weight' / 'k_norm_weight' are not supported when 'kv_cache_layout' is "
                             "'LATENT': DeepSeek normalizes the latent projections in the graph, before absorption.");
    }
    // There is one physical cache, written once with k_scale, so a second scale for the same bytes
    // could only disagree with it. V is dequantized with k_scale.
    if (v_scale != nullptr || v_quant_type != KVQuantizationType::NONE ||
        v_cache_dtype != KVCacheDataType::DEFAULT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'v_scale' and attributes 'v_quant_type' / 'v_cache_dtype' must be unset when "
                             "'kv_cache_layout' is 'LATENT': the value elements are the key elements, so 'k_scale' "
                             "and 'k_cache_dtype' describe both.");
    }
  } else if (value_cache == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'value_cache' is required unless 'kv_cache_layout' is 'LATENT'.");
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
  if (slot_mapping != nullptr) {
    ORT_RETURN_IF_ERROR(CheckSlotMapping(slot_mapping, token_count));
  }

  // Check attention sink and QK-Norm weights
  if (head_sink != nullptr) {
    ORT_RETURN_IF_ERROR(CheckHeadSink(head_sink, num_heads));
  }
  ORT_RETURN_IF_ERROR(CheckQKNormWeights(q_norm_weight, k_norm_weight, head_size));
  if (q_norm_weight != nullptr && !(qk_norm_epsilon > 0.0f)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qk_norm_epsilon must be positive, got ", qk_norm_epsilon);
  }

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

  // Offset (partial) rotary (§12.5). RoPE covers [rotary_offset, rotary_offset + rotary_dim) of
  // each head; MLA rotates only the k_pe suffix. Default 0 is the shipped prefix behavior.
  if (rotary_offset < 0 || rotary_offset % 8 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'rotary_offset' must be non-negative and a multiple of 8, got ", rotary_offset);
  }
  if (rotary_offset + rotary_dim > head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "'rotary_offset' + rotary_dim must not exceed head_size. Got ", rotary_offset, " + ",
                           rotary_dim, " > ", head_size);
  }

  // Check quantized KV cache. LATENT has no value cache to describe, and the block above already
  // required v_scale / v_quant_type to be unset there.
  ORT_RETURN_IF_ERROR(CheckKVCacheQuantization(k_scale, "k_scale", "k_quant_type", k_quant_type,
                                               is_quantized_cache, kv_num_heads, head_size));
  if (!is_latent_kv) {
    ORT_RETURN_IF_ERROR(CheckKVCacheQuantization(v_scale, "v_scale", "v_quant_type", v_quant_type,
                                                 is_quantized_cache, kv_num_heads, v_head_size));
  }
  ORT_RETURN_IF_ERROR(CheckKVCacheDataType(k_cache_dtype, cache_storage_dtype, "k_cache_dtype"));
  ORT_RETURN_IF_ERROR(CheckKVCacheDataType(v_cache_dtype, cache_storage_dtype, "v_cache_dtype"));

  // Optional host-side [max_query_len_bound, max_kv_len_bound]. Only the shape is checked here.
  // The entries are *trusted upper bounds* and cannot be cross-checked against the device tensors
  // they bound without the readback this input exists to remove; see the trust boundary in
  // docs/contrib_ops/cuda/paged_attention.md section 4.7.
  if (attention_metadata != nullptr) {
    const auto& metadata_dims = attention_metadata->Shape().GetDims();
    if (metadata_dims.size() != 1 || metadata_dims[0] != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'attention_metadata' must have shape (2), got ",
                             attention_metadata->Shape().ToString());
    }
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
    output_parameters->v_head_size = v_head_size;
    output_parameters->v_hidden_size = num_heads * v_head_size;
    output_parameters->is_latent_kv = is_latent_kv;
    output_parameters->rotary_offset = rotary_offset;
    output_parameters->block_size = block_size;
    output_parameters->max_num_blocks_per_seq = max_num_blocks_per_seq;
    output_parameters->num_blocks = num_blocks;
    output_parameters->rotary_dim = rotary_dim;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->scale = scale;
    output_parameters->softcap = softcap;
    output_parameters->use_smooth_softmax = head_sink != nullptr;
    output_parameters->use_qk_norm = q_norm_weight != nullptr;
    output_parameters->qk_norm_epsilon = qk_norm_epsilon;
    output_parameters->k_quant_type = k_quant_type;
    output_parameters->v_quant_type = v_quant_type;
  }

  return Status::OK();
}

}  // namespace paged_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
