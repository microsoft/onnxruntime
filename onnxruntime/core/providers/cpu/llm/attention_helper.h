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
  bool transpose_output;      // Whether to transpose the output from BxNxSxH to BxSxNxH

  // Checks the consistency of the parameters.
  void checkParameters() const;
};

// Computes the output shape for attention based on the input tensors and parameters.
Status ComputeOutputShapeForAttention(
    const Tensor* Q,
    const Tensor* K,
    const Tensor* V,
    const Tensor* attn_mask,
    const Tensor* past_key,
    const Tensor* past_value,
    bool is_causal,
    float softcap,
    int softmax_precision,
    attention_helper::QKMatMulOutputMode qk_matmul_output_mode,
    int kv_num_heads,
    int q_num_heads,
    float scale,
    AttentionParameters& parameters,
    std::vector<int64_t>& y_shape,
    std::vector<int64_t>& present_key_shape,
    std::vector<int64_t>& present_value_shape,
    std::vector<int64_t>& output_qk_shape);

}  // namespace attention_helper
}  // namespace onnxruntime
