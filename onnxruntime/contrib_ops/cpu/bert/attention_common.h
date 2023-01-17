// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

enum AttentionMaskType {
  MASK_NONE,            // No mask
  MASK_1D_KEY_SEQ_LEN,  // [batch_size], key sequence length
  MASK_1D_END_START,    // [2 * batch_size] with end positions and start positions
  MASK_2D_DUMMY,        // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  MASK_2D_KEY_PADDING,  // [batch_size, total_sequence_length]
  MASK_3D_ATTENTION,    // [batch_size, sequence_length, total_sequence_length]
  MASK_4D_MEGATRON,     // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  MASK_UNKNOWN
};

struct AttentionParameters {
  int batch_size;
  int sequence_length;
  int kv_sequence_length;     // input sequence length of K or V
  int past_sequence_length;   // sequence length in past state of K or V
  int total_sequence_length;  // total sequence length of K or V
  int max_sequence_length;
  int input_hidden_size;
  int hidden_size;    // hidden size of Q or K
  int head_size;      // hidden size per head of Q or K
  int v_hidden_size;  // hidden size of V
  int v_head_size;    // hidden size per head of V
  int num_heads;
  bool is_unidirectional;
  bool past_present_share_buffer;
  float mask_filter_value;
  AttentionMaskType mask_type;
};

namespace attention {
// Environment variable to enable or disable fused attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedAttention = "ORT_DISABLE_FUSED_ATTENTION";

// Environment variable to enable or disable flash attention. Default is 1 (enabled). Set it to 0 to disable.
constexpr const char* kEnableFlashAttention = "ORT_ENABLE_FLASH_ATTENTION";
}  // namespace attention

}  // namespace contrib
}  // namespace onnxruntime
