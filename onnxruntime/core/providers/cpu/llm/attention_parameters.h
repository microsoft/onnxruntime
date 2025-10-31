// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace attention_helper {

// enum equivalent to the onnx defintion of qk_matmul_output_mode
enum QKMatMulOutputMode {
  kNone = -1,      // No output Q*K
  kQK = 0,         // Output Q*K
  kQKMask = 1,     // Output Q*K + Mask
  kQKSoftCap = 2,  // Output SoftCap(Q*K + Mask)
  kQKSoftMax = 3,  // Output SoftMax(SoftCap(Q*K + Mask))
};

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  /*
   * Attention Parameters
   * MHA: q_num_heads == kv_num_heads -> MHA
   * GQA: q_num_heads > kv_num_heads && q_num_heads % kv_num_heads == 0
   * MQA: q_num_heads > kv_num_heads && kv_num_heads == 1
   */
  bool is_causal;
  int kv_num_heads;  // K.shape[1] or V.shape[1] (4D)
  int q_num_heads;   // Q.shape[1] (4D)
  float scale;
  float softcap;
  int softmax_precision;
  QKMatMulOutputMode qk_matmul_output_mode;

  // From shapes
  int batch_size;             // Q.shape[0], K.shape[0], V.shape[0] (4D)
  int q_sequence_length;      // Q.shape[2] (4D)
  int head_size;              // Q.shape[3] or K.shape[3 (4D)
  int kv_sequence_length;     // K.shape[2] or V.shape[2] (4D)
  int v_head_size;            // V.shape[4] (4D)
  int past_sequence_length;   // pask_key.shape[2] or past_value.shape[2] (4D)
  int total_sequence_length;  // past_sequence_length + kv_sequence_length
  bool transpose_output;      // Whether to transpose the inputs and the outputs from BxNxSxH to BxSxNxH
                              // This covers the case where the inputs are 3D.

  // Checks the consistency of the parameters.
  void checkParameters() const {
    ORT_ENFORCE(batch_size > 0, "Batch size must be greater than 0");
    ORT_ENFORCE(q_sequence_length > 0, "Q sequence length must be greater than 0");
    ORT_ENFORCE(kv_sequence_length > 0, "KV sequence length must be greater than 0");
    ORT_ENFORCE(head_size > 0, "Head size must be greater than 0");
    ORT_ENFORCE(v_head_size > 0, "V head size must be greater than 0");
    ORT_ENFORCE(past_sequence_length >= 0, "Past sequence length must be non-negative");
    ORT_ENFORCE(total_sequence_length > 0, "Total sequence length must be greater than 0");
    ORT_ENFORCE(kv_num_heads > 0, "KV number of heads must be greater than 0");
    ORT_ENFORCE(q_num_heads > 0, "Q number of heads must be greater than 0");
    ORT_ENFORCE(total_sequence_length == past_sequence_length + kv_sequence_length,
                "Total sequence length must be equal to past sequence length plus KV sequence length");
  }
};

}  // namespace attention_helper
}  // namespace onnxruntime
