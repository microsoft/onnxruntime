// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
// Declares enum QKMatMulOutputMode and struct AttentionParameters inside namespace onnxruntime::attention_helper.
namespace attention_helper {

// enum equivalent to the ONNX definition of qk_matmul_output_mode.
//
// IMPORTANT — ORT intentionally LEADS the bundled ONNX submodule on this
// numbering. The bundled ONNX (cmake/external/onnx) is currently v1.21.0,
// which was tagged BEFORE onnx/onnx#7913 was merged upstream; that bundled
// schema therefore still reflects the OLD value mapping (1 = post-mask/bias,
// 2 = post-softcap). This implementation already follows the corrected
// post-#7913 numbering so that ORT will be spec-correct as soon as the next
// ONNX release (v1.22) is bundled, with no behavior change required at
// that point. As a side effect, the as-shipped opset-23 ONNX backend node
// tests under cmake/external/onnx that pin the OLD numbering will fail
// against this implementation until the submodule is bumped — this is
// expected; see PR description for details.
//
// Mode integer numbering follows the ONNX Attention v23/24 pipeline stage
// order, per onnx/onnx#7867 (which corrected the ordering to apply softcap
// BEFORE bias/mask add) and onnx/onnx#7913 (which swapped the integer
// values of modes 1 and 2 to align with the corrected pipeline):
//
//   stage 0: scale * (Q @ K^T)
//   stage 1: softcap (if > 0)
//   stage 2: + attn_bias / + attn_mask
//   stage 3: softmax
//
// TODO(onnx-v1.22): when cmake/external/onnx is bumped to v1.22+ which
// includes ONNX PRs #7867 + #7913, drop the "ORT leads ONNX" caveat above
// and re-enable the corresponding ONNX backend node tests by removing the
// skip blocks in onnxruntime/test/onnx/TestCase.cc::GetBrokenTests() and
// onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc.
enum QKMatMulOutputMode {
  kNone = -1,         // No optional output.
  kQK = 0,            // Raw scale * Q @ K^T (pre-softcap).
  kPostSoftCap = 1,   // Post-softcap, pre-mask/bias.
  kPostMaskBias = 2,  // Post-mask/bias, pre-softmax.
  kPostSoftMax = 3,   // Post-softmax.
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

  // nonpad_kv_seqlen (Opset 24+): per-batch valid KV sequence lengths, shape [batch_size]
  bool has_nonpad_kv_seqlen = false;
  const int64_t* nonpad_kv_seqlen_data = nullptr;

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
