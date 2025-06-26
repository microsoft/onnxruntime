// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace attention_helper {

enum AttentionMaskType {
  MASK_NONE,  // No mask
  MASK_BOOL,  // Boolean mask with shape [batch_size, sequence_length, total_sequence_length] or [batch_size, q_num_heads, q_sequence_length, total_sequence_length
  MASK_ADD,   // Additive mask with shape [batch_size, sequence_length, total_sequence_length] or [batch_size, q_num_heads, q_sequence_length, total_sequence_length
};

enum AttentionType {
  kInvalid = 0,               // Invalid attention type
  kMultiHeadedAttention = 1,  // Multi-headed attention (MHA)
  kGroupQueryAttention = 2,   // Group query attention (GQA)
  kMultiQueryAttention = 3,   // Multi-query attention (MQA)
};

enum QKMatMulOutputMode {
  kNone = 0,  // No output
  kQK = 1,    // Output Q*K
  kQKV = 2,   // Output Q*K and V
};

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  // Attribute from ONNX definition
  bool is_causal;
  int kv_num_heads;  // K.shape[1] or V.shape[1] (4D)
  int q_num_heads;   // Q.shape[1] (4D)
  float scale;
  float softcap;
  int softmax_precision;
  QKMatMulOutputMode qk_matmul_output_mode;
  AttentionMaskType mask_type;

  // From shapes
  int batch_size;            // Q.shape[0], K.shape[0], V.shape[0] (4D)
  int q_sequence_length;     // Q.shape[2] (4D)
  int head_size;             // Q.shape[3] or K.shape[3 (4D)
  int kv_sequence_length;    // K.shape[2] or V.shape[2] (4D)
  int v_head_size;           // V.shape[4] (4D)
  int past_sequence_length;  // pask_key.shape[2] or past_value.shape[2] (4D)

  AttentionType getAttentionType() const {
    if (q_num_heads == kv_num_heads) {
      return AttentionType::kMultiHeadedAttention;
    } else if (q_num_heads > kv_num_heads && q_num_heads % kv_num_heads == 0) {
      return AttentionType::kGroupQueryAttention;
    } else if (q_num_heads > kv_num_heads && kv_num_heads == 1) {
      return AttentionType::kMultiQueryAttention;
    } else {
      return AttentionType::kInvalid;
    }
  }
};

}  // namespace attention_helper
}  // namespace onnxruntime
