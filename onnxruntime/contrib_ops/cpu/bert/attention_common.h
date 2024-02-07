// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

enum AttentionMaskType {
  MASK_NONE,                  // No mask
  MASK_1D_KEY_SEQ_LEN,        // [batch_size], key sequence length
  MASK_1D_END_START,          // [2 * batch_size] with end positions and start positions
  MASK_1D_KEY_SEQ_LEN_START,  // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
                              // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
                              // key_start[batch_size - 1], key_end[batch_size - 1]]
  MASK_2D_DUMMY,              // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  MASK_2D_KEY_PADDING,        // [batch_size, total_sequence_length]
  MASK_3D_ATTENTION,          // [batch_size, sequence_length, total_sequence_length]
  MASK_4D_MEGATRON,           // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  MASK_UNKNOWN
};

enum AttentionQkvFormat {
  UNKNOWN,               // enum value not set, or depends on qkv projection implementation details
  Q_K_V_BNSH,            // for non-packed qkv, permuted
  Q_K_V_BSNH,            // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  QKV_BSN3H,             // for TRT fused attention, qkv are packed
  Q_K_V_BNSH_QKV_BS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  Q_KV_BSNH_BSN2H,       // for TRT fused cross attention, kv are packed
  Q_K_V_TNH,             // for memory efficient attention, qkv are not packed, and paddings are removed.
  QKV_TN3H,              // for TRT fused attention, qkv are packed and paddings are removed
};

enum AttentionKernelType {
  AttentionKernel_Unfused,
  AttentionKernel_TrtFusedAttention,
  AttentionKernel_TrtFlashAttention,
  AttentionKernel_TrtFusedCrossAttention,
  AttentionKernel_CutlassMemoryEfficientAttention,
  AttentionKernel_FlashAttention,
  AttentionKernel_Default
};

// Parameters deduced from node attributes and inputs/outputs.
struct AttentionParameters {
  int batch_size;
  int sequence_length;
  int kv_sequence_length;     // input sequence length of K or V
  int past_sequence_length;   // sequence length in past state of K or V
  int total_sequence_length;  // total sequence length of K or V
  int max_sequence_length;    // max sequence length from 4D mask
  int input_hidden_size;      // first dimension of weights for input projection
  int hidden_size;            // hidden size of Q or K
  int head_size;              // hidden size per head of Q or K
  int v_hidden_size;          // hidden size of V
  int v_head_size;            // hidden size per head of V
  int num_heads;
  int num_splits;
  int rotary_embedding;
  bool is_unidirectional;
  bool past_present_share_buffer;
  bool do_rotary;
  bool broadcast_res_pos_bias;
  bool pass_past_in_kv;
  float mask_filter_value;
  float scale;
  bool use_tf32;
  AttentionMaskType mask_type;
  AttentionQkvFormat qkv_format;
};

// Parameters deduced from node attributes and inputs/outputs.
struct PackedAttentionParameters {
  int batch_size;
  int sequence_length;
  int input_hidden_size;  // hidden size of input
  int hidden_size;        // hidden size of Q or K
  int head_size;          // hidden size per head of Q or K
  int v_hidden_size;      // hidden size of V
  int v_head_size;        // hidden size per head of V
  int num_heads;
  float scale;
  int token_count;
  bool has_relative_position_bias;
  bool broadcast_res_pos_bias;
  bool use_tf32;
};

// Parameters deduced from node attributes and inputs/outputs.
struct GroupQueryAttentionParameters {
  int batch_size;
  int sequence_length;          // sequence length of input query, key, value
  int seqlen_past_kv_cache;     // sequence length of past kv tensor
  int seqlen_present_kv_cache;  // sequence length of present kv tensor
  int hidden_size;
  int num_heads;
  int head_size;
  int kv_hidden_size;
  int kv_num_heads;
  int num_splits;          // number of splits for splitkv
  bool is_unidirectional;  // causal
  int local_window_size;
  bool kv_share_buffer;
  bool is_packed_qkv;
  bool is_prompt;  // determines if seqlens_k is past or kv sequence length tensor
  bool do_rotary;
  bool rotary_interleaved;
  float scale;
  AttentionQkvFormat qkv_format;
  AttentionQkvFormat past_kv_format;
  int zeros_count;
  int* zero_ptr;
};

namespace attention {
// Environment variable to enable or disable TRT fused self attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedSelfAttention = "ORT_DISABLE_FUSED_ATTENTION";

// Environment variable to enable or disable fused cross attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedCrossAttention = "ORT_DISABLE_FUSED_CROSS_ATTENTION";

// Environment variable to enable or disable TRT fused causal attention kernels. Default is 0 (disabled).
// Note that those causal attention kernels use fp16 accumulation. There is potential accuracy drop using those kernels.
constexpr const char* kEnableFusedCausalAttention = "ORT_ENABLE_FUSED_CAUSAL_ATTENTION";

// Environment variable to enable or disable TRT flash attention. This applies to both self and causal attention. Default is 0 (enabled).
constexpr const char* kDisableTrtFlashAttention = "ORT_DISABLE_TRT_FLASH_ATTENTION";

// Environment variable to enable or disable cutlass memory efficient attention. Default is 0 (enabled).
constexpr const char* kDisableMemoryEfficientAttention = "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION";

// Environment variable to enable or disable flash attention. Default is 0 (enabled).
constexpr const char* kDisableFlashAttention = "ORT_DISABLE_FLASH_ATTENTION";

// Minimum sequence length to enable memory efficient attention in FP32.
constexpr int kMinSeqLenForMemoryEfficientAttentionFp32 = 256;

// Minimum sequence length to prefer flash attention when input format is packed QKV for MultiHeadAttention
constexpr const char* kMinSeqLenForFlashAttentionPackedQKV = "ORT_MIN_SEQ_LEN_FLASH_ATTENTION_PACKED_QKV";
// Default value for the above setting.
constexpr int kDefaultMinSeqLenForFlashAttentionPackedQKV = 513;

// Environment variable to enable loading more KV data in flight in
// DecoderMaskedMultiHeadAttention/DecoderMaskedSelfAttention kernels
constexpr const char* kDecoderMaskedAttentionLoadKVDataInFlight = "ORT_DECODER_MASKED_ATTENTION_LOAD_KV_DATA_IN_FLIGHT";

}  // namespace attention

}  // namespace contrib
}  // namespace onnxruntime
