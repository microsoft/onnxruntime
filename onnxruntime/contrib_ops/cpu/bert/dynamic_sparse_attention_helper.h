// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_parameters.h"
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace dynamic_sparse_attention_helper {

template <typename T>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* value,
                   const T* past_key,
                   const T* past_value,
                   const T* auxiliary_key,
                   const T* auxiliary_value,
                   const T* selected_indices,
                   const T* selected_counts,
                   const T* seqlens_k,
                   const T* total_sequence_length,
                   const T* cos_cache,
                   const T* sin_cache,
                   const T* position_ids,
                   const T* q_norm_weight,
                   const T* k_norm_weight,
                   const T* head_sink,
                   int num_heads,
                   int kv_num_heads,
                   int local_window_size,
                   int rotary_offset,
                   bool do_rotary,
                   bool auxiliary_kv_shared,
                   DynamicSparseAttentionMode attention_mode,
                   DynamicSparseAttentionKvSource selected_kv_source,
                   float scale,
                   float qk_norm_epsilon,
                   DynamicSparseAttentionParameters& parameters) {
  ORT_RETURN_IF_NOT(query != nullptr, "DynamicSparseAttention: query is required.");
  ORT_RETURN_IF_NOT(num_heads > 0 && kv_num_heads > 0 && num_heads % kv_num_heads == 0,
                    "DynamicSparseAttention: num_heads must be a positive multiple of kv_num_heads.");

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3,
                    "DynamicSparseAttention: query must have shape [B, S, D].");
  const int64_t batch_size_64 = query_shape[0];
  const int64_t sequence_length_64 = query_shape[1];
  const int64_t query_width_64 = query_shape[2];
  ORT_RETURN_IF_NOT(batch_size_64 > 0 && sequence_length_64 > 0 && query_width_64 > 0,
                    "DynamicSparseAttention: query dimensions must be positive.");
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max() &&
                        sequence_length_64 <= std::numeric_limits<int>::max() &&
                        query_width_64 <= std::numeric_limits<int>::max(),
                    "DynamicSparseAttention: query dimensions exceed CUDA kernel limits.");

  const int batch_size = static_cast<int>(batch_size_64);
  const int sequence_length = static_cast<int>(sequence_length_64);
  const bool is_packed_qkv = key == nullptr;
  ORT_RETURN_IF_NOT((key == nullptr) == (value == nullptr),
                    "DynamicSparseAttention: key and value must either both be present or both be absent.");

  int head_size = 0;
  int query_hidden_size = 0;
  int kv_hidden_size = 0;
  if (is_packed_qkv) {
    const int64_t packed_heads = static_cast<int64_t>(num_heads) + 2LL * kv_num_heads;
    ORT_RETURN_IF_NOT(query_width_64 % packed_heads == 0,
                      "DynamicSparseAttention: packed query width must be divisible by num_heads + 2 * kv_num_heads.");
    head_size = static_cast<int>(query_width_64 / packed_heads);
    query_hidden_size = static_cast<int>(static_cast<int64_t>(num_heads) * head_size);
    kv_hidden_size = static_cast<int>(static_cast<int64_t>(kv_num_heads) * head_size);
  } else {
    ORT_RETURN_IF_NOT(query_width_64 % num_heads == 0,
                      "DynamicSparseAttention: query width must be divisible by num_heads.");
    head_size = static_cast<int>(query_width_64 / num_heads);
    query_hidden_size = static_cast<int>(query_width_64);
    const int64_t kv_hidden_size_64 = static_cast<int64_t>(kv_num_heads) * head_size;
    ORT_RETURN_IF_NOT(kv_hidden_size_64 <= std::numeric_limits<int>::max(),
                      "DynamicSparseAttention: KV hidden size exceeds CUDA kernel limits.");
    kv_hidden_size = static_cast<int>(kv_hidden_size_64);

    const auto& key_shape = key->Shape();
    const auto& value_shape = value->Shape();
    ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 3 && value_shape.NumDimensions() == 3,
                      "DynamicSparseAttention: key and value must have shape [B, S, kv_num_heads * H].");
    ORT_RETURN_IF_NOT(key_shape[0] == batch_size && value_shape[0] == batch_size &&
                          key_shape[1] == sequence_length && value_shape[1] == sequence_length,
                      "DynamicSparseAttention: query, key, and value batch/sequence dimensions must match.");
    ORT_RETURN_IF_NOT(key_shape[2] == kv_hidden_size && value_shape[2] == kv_hidden_size,
                      "DynamicSparseAttention: key and value widths must equal kv_num_heads * head_size.");
  }
  ORT_RETURN_IF_NOT(head_size > 0, "DynamicSparseAttention: head_size must be positive.");

  ORT_RETURN_IF_NOT((past_key == nullptr) == (past_value == nullptr),
                    "DynamicSparseAttention: past_key and past_value must either both be present or both be absent.");
  int past_cache_capacity = 0;
  if (past_key != nullptr) {
    const auto& key_shape = past_key->Shape();
    const auto& value_shape = past_value->Shape();
    ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 4 && value_shape.NumDimensions() == 4,
                      "DynamicSparseAttention: past_key and past_value must have BNSH layout.");
    ORT_RETURN_IF_NOT(key_shape == value_shape,
                      "DynamicSparseAttention: past_key and past_value must have identical shapes.");
    ORT_RETURN_IF_NOT(key_shape[0] == batch_size && key_shape[1] == kv_num_heads && key_shape[3] == head_size,
                      "DynamicSparseAttention: past cache shape must be [B, kv_num_heads, capacity, H].");
    ORT_RETURN_IF_NOT(key_shape[2] >= 0 && key_shape[2] <= std::numeric_limits<int>::max(),
                      "DynamicSparseAttention: past cache capacity exceeds CUDA kernel limits.");
    past_cache_capacity = static_cast<int>(key_shape[2]);
  }

  ORT_RETURN_IF_NOT(total_sequence_length != nullptr &&
                        onnxruntime::IsScalarOr1ElementVector(total_sequence_length),
                    "DynamicSparseAttention: total_sequence_length must be a scalar or one-element vector.");
  const int total_length = *total_sequence_length->template Data<int32_t>();
  ORT_RETURN_IF_NOT(total_length >= sequence_length,
                    "DynamicSparseAttention: total_sequence_length must be at least sequence length.");
  ORT_RETURN_IF_NOT(past_key != nullptr || total_length == sequence_length,
                    "DynamicSparseAttention: total_sequence_length must equal sequence length when no past cache "
                    "is provided.");
  ORT_RETURN_IF_NOT(past_key == nullptr || total_length <= past_cache_capacity,
                    "DynamicSparseAttention: total_sequence_length must not exceed the past cache capacity.");
  const int cache_capacity = past_key == nullptr ? total_length : past_cache_capacity;

  ORT_RETURN_IF_NOT(seqlens_k != nullptr && seqlens_k->Shape().NumDimensions() == 1 &&
                        seqlens_k->Shape()[0] == batch_size,
                    "DynamicSparseAttention: seqlens_k must have shape [B].");
  ORT_RETURN_IF_NOT(selected_counts != nullptr && selected_counts->Shape().NumDimensions() == 1 &&
                        selected_counts->Shape()[0] == static_cast<int64_t>(batch_size) * sequence_length,
                    "DynamicSparseAttention: selected_counts must have shape [B * S].");
  ORT_RETURN_IF_NOT(selected_indices != nullptr && selected_indices->Shape().NumDimensions() == 2 &&
                        selected_indices->Shape()[0] == static_cast<int64_t>(batch_size) * sequence_length,
                    "DynamicSparseAttention: selected_indices must have shape [B * S, max_selected].");
  ORT_RETURN_IF_NOT(selected_indices->Shape()[1] >= 0 &&
                        selected_indices->Shape()[1] <= std::numeric_limits<int>::max(),
                    "DynamicSparseAttention: max_selected exceeds CUDA kernel limits.");

  int auxiliary_sequence_length = 0;
  ORT_RETURN_IF_NOT(auxiliary_key != nullptr || auxiliary_value == nullptr,
                    "DynamicSparseAttention: auxiliary_value cannot be present without auxiliary_key.");
  ORT_RETURN_IF_NOT(auxiliary_value != nullptr || auxiliary_key == nullptr || auxiliary_kv_shared,
                    "DynamicSparseAttention: auxiliary_value may be omitted only when auxiliary_kv_shared is 1.");
  if (auxiliary_key != nullptr) {
    const auto& key_shape = auxiliary_key->Shape();
    ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 4,
                      "DynamicSparseAttention: auxiliary_key must have BNSH layout.");
    if (auxiliary_value != nullptr) {
      ORT_RETURN_IF_NOT(auxiliary_value->Shape().NumDimensions() == 4 &&
                            key_shape == auxiliary_value->Shape(),
                        "DynamicSparseAttention: auxiliary_key and auxiliary_value must have identical BNSH shapes.");
    }
    ORT_RETURN_IF_NOT(key_shape[0] == batch_size && key_shape[1] == kv_num_heads &&
                          key_shape[3] == head_size,
                      "DynamicSparseAttention: auxiliary cache shape is incompatible with the attributes.");
    ORT_RETURN_IF_NOT(key_shape[2] >= 0 && key_shape[2] <= std::numeric_limits<int>::max(),
                      "DynamicSparseAttention: auxiliary sequence length exceeds CUDA kernel limits.");
    auxiliary_sequence_length = static_cast<int>(key_shape[2]);
  }
  ORT_RETURN_IF_NOT(selected_kv_source != DynamicSparseAttentionKvSource::kAuxiliary ||
                        auxiliary_key != nullptr,
                    "DynamicSparseAttention: auxiliary KV inputs are required when selected_kv_source is auxiliary.");
  ORT_RETURN_IF_NOT(
      (attention_mode == DynamicSparseAttentionMode::kSelectedOnly &&
       selected_kv_source == DynamicSparseAttentionKvSource::kMain) ||
          (attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected &&
           selected_kv_source == DynamicSparseAttentionKvSource::kAuxiliary),
      "DynamicSparseAttention: supported mode/source combinations are selected_only with main and "
      "local_plus_selected with auxiliary.");
  ORT_RETURN_IF_NOT(selected_kv_source != DynamicSparseAttentionKvSource::kMain ||
                        auxiliary_key == nullptr,
                    "DynamicSparseAttention: auxiliary KV inputs are not allowed when selected_kv_source is main.");

  ORT_RETURN_IF_NOT(attention_mode != DynamicSparseAttentionMode::kLocalPlusSelected ||
                        local_window_size > 0,
                    "DynamicSparseAttention: local_plus_selected requires local_window_size > 0.");
  ORT_RETURN_IF_NOT(std::isfinite(scale) && scale >= 0.0f,
                    "DynamicSparseAttention: scale must be finite and nonnegative.");
  ORT_RETURN_IF_NOT(std::isfinite(qk_norm_epsilon) && qk_norm_epsilon > 0.0f,
                    "DynamicSparseAttention: qk_norm_epsilon must be finite and positive.");

  ORT_RETURN_IF_NOT((q_norm_weight == nullptr) == (k_norm_weight == nullptr),
                    "DynamicSparseAttention: q_norm_weight and k_norm_weight must both be present or absent.");
  if (q_norm_weight != nullptr) {
    ORT_RETURN_IF_NOT(q_norm_weight->Shape().NumDimensions() == 1 &&
                          k_norm_weight->Shape().NumDimensions() == 1 &&
                          q_norm_weight->Shape()[0] == head_size &&
                          k_norm_weight->Shape()[0] == head_size,
                      "DynamicSparseAttention: Q/K RMSNorm weights must have shape [H].");
  }
  if (head_sink != nullptr) {
    ORT_RETURN_IF_NOT(head_sink->Shape().NumDimensions() == 1 && head_sink->Shape()[0] == num_heads,
                      "DynamicSparseAttention: head_sink must have shape [num_heads].");
  }
  if (position_ids != nullptr) {
    ORT_RETURN_IF_NOT(position_ids->Shape().NumDimensions() == 2 &&
                          position_ids->Shape()[0] == batch_size &&
                          position_ids->Shape()[1] == sequence_length,
                      "DynamicSparseAttention: position_ids must have shape [B, S].");
  }

  int rotary_dim = 0;
  int rotary_max_position = 0;
  ORT_RETURN_IF_NOT((cos_cache == nullptr) == (sin_cache == nullptr),
                    "DynamicSparseAttention: cos_cache and sin_cache must both be present or absent.");
  if (do_rotary) {
    ORT_RETURN_IF_NOT(cos_cache != nullptr,
                      "DynamicSparseAttention: rotary caches are required when do_rotary is 1.");
  }
  if (cos_cache != nullptr) {
    const auto& cos_shape = cos_cache->Shape();
    const auto& sin_shape = sin_cache->Shape();
    ORT_RETURN_IF_NOT(cos_shape.NumDimensions() == 2 && cos_shape == sin_shape &&
                          cos_shape[0] > 0 && cos_shape[0] <= std::numeric_limits<int>::max() &&
                          cos_shape[1] > 0 && cos_shape[1] <= std::numeric_limits<int>::max() / 2,
                      "DynamicSparseAttention: rotary caches must have identical rank-2 shapes.");
    rotary_dim = static_cast<int>(cos_shape[1] * 2);
    rotary_max_position = static_cast<int>(cos_shape[0]);
  }
  ORT_RETURN_IF_NOT(rotary_offset >= 0 && rotary_offset % 8 == 0,
                    "DynamicSparseAttention: rotary_offset must be nonnegative and a multiple of 8.");
  ORT_RETURN_IF_NOT(static_cast<int64_t>(rotary_offset) + rotary_dim <= head_size,
                    "DynamicSparseAttention: rotary_offset + rotary_dim must not exceed head_size.");

  parameters.batch_size = batch_size;
  parameters.sequence_length = sequence_length;
  parameters.num_heads = num_heads;
  parameters.kv_num_heads = kv_num_heads;
  parameters.head_size = head_size;
  parameters.query_hidden_size = query_hidden_size;
  parameters.kv_hidden_size = kv_hidden_size;
  parameters.cache_capacity = cache_capacity;
  parameters.past_cache_capacity = past_cache_capacity;
  parameters.total_sequence_length = total_length;
  parameters.auxiliary_sequence_length = auxiliary_sequence_length;
  parameters.max_selected = static_cast<int>(selected_indices->Shape()[1]);
  parameters.rotary_dim = do_rotary ? rotary_dim : 0;
  parameters.rotary_max_position = rotary_max_position;
  parameters.rotary_offset = rotary_offset;
  parameters.local_window_size = local_window_size;
  parameters.scale = scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(head_size)) : scale;
  parameters.qk_norm_epsilon = qk_norm_epsilon;
  parameters.is_packed_qkv = is_packed_qkv;
  parameters.do_rotary = do_rotary;
  parameters.use_qk_norm = q_norm_weight != nullptr;
  parameters.auxiliary_kv_shared = auxiliary_kv_shared;
  parameters.attention_mode = attention_mode;
  parameters.selected_kv_source = selected_kv_source;
  return Status::OK();
}

}  // namespace dynamic_sparse_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
