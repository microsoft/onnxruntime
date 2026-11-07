// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace multihead_latent_attention_helper {

// Checks the shapes of the inputs and total_sequence_length, and fills `parameters`. The rules on lengths are
// GroupQueryAttention's with kv_num_heads = 1:
//   query                  (B, S, num_heads * head_size)
//   key                    (B, S, head_size)
//   past_kv                (B, 1, L, head_size), optional
//   seqlens_k              (B), total sequence length (past + new) - 1 of each batch entry
//   total_sequence_length  scalar or (1)
// A step with S == total_sequence_length is a first prompt. Otherwise S > 1 requires B == 1, and a cache shorter
// than total_sequence_length must grow by exactly S. present_kv has length max(total_sequence_length, L).
template <typename T = Tensor>
Status CheckInputs(const T* query,
                   const T* key,
                   const T* past_kv,
                   const T* seqlens_k,
                   const T* total_seqlen,
                   int num_heads,
                   int v_head_size,
                   int v_offset,
                   float scale,
                   MultiHeadLatentAttentionParameters& parameters) {
  if (num_heads <= 0 || v_head_size <= 0 || v_offset < 0 || !(scale > 0.0f)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads, v_head_size and scale must be positive and v_offset non-negative. Got num_heads=",
                           num_heads, ", v_head_size=", v_head_size, ", v_offset=", v_offset, ", scale=", scale);
  }

  const auto& query_dims = query->Shape().GetDims();
  if (query_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_dims.size());
  }
  const int64_t batch_size = query_dims[0];
  const int64_t sequence_length = query_dims[1];
  const int64_t hidden_size = query_dims[2];
  if (hidden_size % num_heads != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query hidden size (", hidden_size,
                           ") must be divisible by num_heads (", num_heads, ")");
  }
  const int64_t head_size = hidden_size / num_heads;
  if (static_cast<int64_t>(v_offset) + v_head_size > head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "v_offset + v_head_size (", v_offset, " + ", v_head_size,
                           ") must not exceed head_size (", head_size, ")");
  }

  const auto& key_dims = key->Shape().GetDims();
  if (key_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_dims.size());
  }
  if (key_dims[0] != batch_size || key_dims[1] != sequence_length || key_dims[2] != head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have shape (", batch_size, ", ",
                           sequence_length, ", ", head_size, "), got ", key->Shape());
  }

  int64_t past_sequence_length = 0;
  if (past_kv != nullptr) {
    const auto& past_dims = past_kv->Shape().GetDims();
    if (past_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past_kv' is expected to have 4 dimensions, got ",
                             past_dims.size());
    }
    if (past_dims[0] != batch_size || past_dims[1] != 1 || past_dims[3] != head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past_kv' is expected to have shape (", batch_size,
                             ", 1, past_sequence_length, ", head_size, "), got ", past_kv->Shape());
    }
    past_sequence_length = past_dims[2];
  }

  // As in GroupQueryAttention, unit dimensions around (batch_size) are accepted.
  if (seqlens_k->Shape().NumDimensions() == 0 || seqlens_k->Shape().Size() != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "seqlens_k must have shape (batch_size) with batch_size = ",
                           batch_size, ", got ", seqlens_k->Shape());
  }
  for (int64_t dim : seqlens_k->Shape().GetDims()) {
    if (dim != 1 && dim != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "seqlens_k must have shape (batch_size) with batch_size = ",
                             batch_size, ", got ", seqlens_k->Shape());
    }
  }

  if (!onnxruntime::IsScalarOr1ElementVector(total_seqlen)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "total_sequence_length tensor must be of one element.");
  }
  const int64_t total_sequence_length = *total_seqlen->template Data<int32_t>();
  if (total_sequence_length <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "total_sequence_length must be positive, got ",
                           total_sequence_length);
  }
  if (total_sequence_length < sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "total_sequence_length (", total_sequence_length,
                           ") must be at least sequence_length (", sequence_length, ")");
  }
  if (total_sequence_length > past_sequence_length &&
      total_sequence_length - sequence_length != past_sequence_length) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "total_sequence_length must equal past_sequence_length + sequence_length for a growing cache, or be no "
        "greater than past_sequence_length for a preallocated one. Got total_sequence_length=",
        total_sequence_length, ", past_sequence_length=", past_sequence_length,
        ", sequence_length=", sequence_length);
  }

  const bool is_first_prompt = sequence_length == total_sequence_length;
  if (!is_first_prompt) {
    if (sequence_length == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_length must be positive.");
    }
    if (sequence_length > 1 && batch_size != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "batch_size must be 1 when sequence_length > 1 and past context is given.");
    }
  }

  const int64_t present_sequence_length = std::max(total_sequence_length, past_sequence_length);
  constexpr int64_t kMaxInt = std::numeric_limits<int32_t>::max();
  if (batch_size > kMaxInt || sequence_length * num_heads > kMaxInt || hidden_size > kMaxInt ||
      present_sequence_length > kMaxInt || static_cast<int64_t>(num_heads) * v_head_size > kMaxInt) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MultiHeadLatentAttention dimensions must fit in int32.");
  }

  parameters.batch_size = static_cast<int>(batch_size);
  parameters.sequence_length = static_cast<int>(sequence_length);
  parameters.kv_sequence_length = static_cast<int>(sequence_length);
  parameters.past_sequence_length = static_cast<int>(past_sequence_length);
  parameters.total_sequence_length = static_cast<int>(total_sequence_length);
  parameters.present_sequence_length = static_cast<int>(present_sequence_length);
  parameters.hidden_size = static_cast<int>(hidden_size);
  parameters.head_size = static_cast<int>(head_size);
  parameters.num_heads = num_heads;
  parameters.v_head_size = v_head_size;
  parameters.v_hidden_size = num_heads * v_head_size;
  parameters.v_offset = v_offset;
  parameters.scale = scale;
  parameters.is_first_prompt = is_first_prompt;
  parameters.is_unidirectional = true;
  parameters.qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  return Status::OK();
}

// Resolves the rows of batch entry b from seqlens_k[b], with GroupQueryAttention's rules:
//   total_rows = seqlens_k[b] + 1 rows are valid in the cache after the step;
//   past_rows rows were there before: 0 on a first prompt (which may be right-padded, total_rows < S),
//   total_rows - S otherwise.
// The S new rows go to cache rows [past_rows, past_rows + S).
inline Status GetBatchRows(int32_t seqlens_k_b,
                           int batch_index,
                           const MultiHeadLatentAttentionParameters& parameters,
                           int& past_rows,
                           int& total_rows) {
  if (seqlens_k_b < 0 || seqlens_k_b >= parameters.present_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "seqlens_k[", batch_index, "] = ", seqlens_k_b,
                           " is out of range [0, ", parameters.present_sequence_length, ")");
  }
  total_rows = seqlens_k_b + 1;
  if (parameters.is_first_prompt) {
    past_rows = 0;
    return Status::OK();
  }
  if (total_rows < parameters.sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "seqlens_k[", batch_index, "] = ", seqlens_k_b,
                           " is too small for sequence_length ", parameters.sequence_length);
  }
  past_rows = total_rows - parameters.sequence_length;
  if (past_rows > parameters.past_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "seqlens_k[", batch_index, "] = ", seqlens_k_b,
                           " requires ", past_rows, " past rows, which exceeds the past_kv sequence length ",
                           parameters.past_sequence_length);
  }
  return Status::OK();
}

}  // namespace multihead_latent_attention_helper
}  // namespace contrib
}  // namespace onnxruntime
