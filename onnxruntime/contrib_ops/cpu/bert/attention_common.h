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

enum AttentionQkvFormat {
  Q_K_V_BNSH,            // for unfused attention
  Q_K_V_BSNH,            // for memory efficient attention, or format of query, key and value for MultiHeadAttention
  QKV_BSN3H,             // for TRT fused attention, qkv are packed
  Q_K_V_BNSH_QKV_BS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  Q_KV_BSNH_BSN2H,       // for TRT fused cross attention, kv are packed
};

enum AttentionKernelType{
  AttentionKernel_Unfused,
  AttentionKernel_TrtFusedAttention,
  AttentionKernel_TrtFlashAttention,
  AttentionKernel_TrtFusedCrossAttention,
  AttentionKernel_CutlassMemoryEfficientAttention,
  AttentionKernel_Default
};

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  int batch_size;
  int sequence_length;
  int kv_sequence_length;            // input sequence length of K or V
  int past_sequence_length;          // sequence length in past state of K or V
  int total_sequence_length;         // total sequence length of K or V
  int max_sequence_length;           // max sequence length from 4D mask
  int input_hidden_size;             // first dimension of weights for input projection
  int hidden_size;                   // hidden size of Q or K
  int head_size;                     // hidden size per head of Q or K
  int v_hidden_size;                 // hidden size of V
  int v_head_size;                   // hidden size per head of V
  int num_heads;
  bool is_unidirectional;
  bool past_present_share_buffer;
  float mask_filter_value;
  float scale;
  AttentionMaskType mask_type;
};

namespace attention {
// Environment variable to enable or disable fused self/causal attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedAttention = "ORT_DISABLE_FUSED_ATTENTION";

// Environment variable to enable or disable fused cross attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedCrossAttention = "ORT_DISABLE_FUSED_CROSS_ATTENTION";

// Environment variable to enable or disable TRT flash attention. Default is 0 (enabled).
constexpr const char* kDisableTrtFlashAttention = "ORT_DISABLE_TRT_FLASH_ATTENTION";

// Environment variable to enable or disable cutlass memory efficient attention. Default is 0 (enabled).
constexpr const char* kDisableMemoryEfficientAttention = "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION";

// Minimum sequence length to enable memory efficient attention in FP32.
constexpr int kMinSequenceLengthForMemoryEfficientAttentionFp32 = 256;

}  // namespace attention

}  // namespace contrib
}  // namespace onnxruntime
