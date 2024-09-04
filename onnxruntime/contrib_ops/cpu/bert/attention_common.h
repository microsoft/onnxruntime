// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <gsl/gsl>

namespace onnxruntime {
namespace contrib {

enum AttentionType {
  kAttention,
  kMultiHeadAttention,
  kDecoderMaskedMultiHeadAttention,
};

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
  Q_K_V_BSNH_BNSH_BNSH,  // for cross attention, k and v are permuted
  Q_K_V_BNSH_QKV_BS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  Q_K_V_TNH,             // for memory efficient attention, qkv are not packed, and paddings are removed.
  Q_KV_BSNH_BSN2H,       // for TRT fused cross attention, kv are packed
  QKV_BSN3H,             // for TRT fused attention, qkv are packed
  QKV_BS3NH,             // for DecoderMaskedMultiHeadAttention, qkv are packed
  QKV_TN3H,              // for TRT fused attention, qkv are packed and paddings are removed
};

enum AttentionKernelType {
  AttentionKernel_Unfused,
  AttentionKernel_TrtFusedAttention,
  AttentionKernel_TrtFlashAttention,
  AttentionKernel_TrtFusedCrossAttention,
  AttentionKernel_CutlassMemoryEfficientAttention,
  AttentionKernel_FlashAttention,
  AttentionKernel_CudnnFlashAttention,
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
  bool broadcast_attn_bias_dim_0;
  bool broadcast_attn_bias_dim_1;
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
  bool broadcast_attn_bias_dim_0;
  bool broadcast_attn_bias_dim_1;
  bool use_tf32;
};

// Parameters deduced from node attributes and inputs/outputs.
struct GroupQueryAttentionParameters {
  int batch_size;
  int sequence_length;          // sequence length of input query, key, value
  int seqlen_past_kv_cache;     // sequence length of past kv tensor
  int seqlen_present_kv_cache;  // sequence length of present kv tensor
  int total_sequence_length;    // maximum total sequence length (past_sequence_length + sequence_length) among keys
  int hidden_size;
  int num_heads;
  int head_size;
  int kv_hidden_size;
  int kv_num_heads;
  int num_splits;          // number of splits for splitkv
  int rotary_dim;          // rotary embedding dimension
  bool is_unidirectional;  // causal
  int local_window_size;
  bool kv_share_buffer;
  bool is_packed_qkv;
  bool is_interactive;  // indicates whether we have past context and seqlen > 1
  bool is_prompt;       // indicates whether this is first decoding step
  bool do_rotary;
  bool rotary_interleaved;
  bool use_smooth_softmax;
  float scale;
  float softcap;
  AttentionQkvFormat qkv_format;
  AttentionQkvFormat past_kv_format;
  int zeros_count;
  int* zero_ptr;
};

// Parameters for sparse attention.
struct SparseAttentionParameters {
  int batch_size;                  // batch size
  int sequence_length;             // sequence length of input query, key, value
  int hidden_size;                 // hidden size of query
  int num_heads;                   // number of heads of query
  int head_size;                   // hidden size per head of query, key or value
  int kv_hidden_size;              // hidden size of key or value
  int kv_num_heads;                // number of heads of key or value
  bool do_rotary;                  // whether to use rotary embedding
  bool rotary_interleaved;         // whether to use interleaved rotary embedding
  int rotary_dim;                  // rotary embedding dimension
  int sparse_block_size;           // block size for sparse attention
  int num_sparse_layout;           // number of sparse layout
  int stride_col_indices;          // shape of block_col_indices is [num_sparse_layout, stride_col_indices]
  int stride_row_indices;          // shape of block_row_indices is [num_sparse_layout, stride_row_indices]
  float scale;                     // scaling factor applied prior to softmax
  bool is_packed_qkv;              // whether qkv is packed
  int total_sequence_length;       // maximum total sequence length (past_sequence_length + sequence_length) among keys
  int max_sequence_length;         // max sequence length for sparse layout
  int max_rotary_sequence_length;  // max sequence length for rotary cos/sin cache
  int max_cache_sequence_length;   // max sequence length for kv cache buffer
  bool past_present_share_buffer;  // whether past_key and present_key share buffer, so is past_value and present_value
};

constexpr bool LAYOUT_BSNH = false;
constexpr bool LAYOUT_BNSH = true;

namespace sparse_attention {
// Environment variable to enable or disable sparse attention v1 kernel. Default is 0 (enabled).
constexpr const char* kDisableSparseAttentionV1 = "ORT_DISABLE_SPARSE_ATTENTION_V1";
}  // namespace sparse_attention

namespace attention {

enum class AttentionBackend : int {
  FLASH_ATTENTION = 1,
  EFFICIENT_ATTENTION = 2,
  TRT_FUSED_ATTENTION = 4,
  CUDNN_FLASH_ATTENTION = 8,  // reserved for cuDNN flash attention.
  MATH = 16,                  // unfused kernel cannot be disabled right now.

  // The following kernels might be deprecated in the future.
  TRT_FLASH_ATTENTION = 32,
  TRT_CROSS_ATTENTION = 64,
  TRT_CAUSAL_ATTENTION = 128,
};

// Environment variable to enable debug information of attention kernel to be printed. Default is 0 (disabled).
constexpr const char* kEnableAttentionKernelDebugInfo = "ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO";

// Environment variable to enable or disable TRT fused self attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedSelfAttention = "ORT_DISABLE_FUSED_ATTENTION";

// Environment variable to enable or disable fused cross attention kernel. Default is 0 (enabled).
constexpr const char* kDisableFusedCrossAttention = "ORT_DISABLE_FUSED_CROSS_ATTENTION";

// Environment variable to enable or disable TRT fused causal attention kernels. Default is 0 (disabled).
// Note that those causal attention kernels use fp16 accumulation. There is potential accuracy drop using those kernels.
constexpr const char* kEnableFusedCausalAttention = "ORT_ENABLE_FUSED_CAUSAL_ATTENTION";

// Environment variable to enable or disable cuDNN flash attention.
constexpr const char* kEnableCudnnFlashAttention = "ORT_ENABLE_CUDNN_FLASH_ATTENTION";

// Environment variable to enable or disable TRT flash attention. This applies to both self and causal attention. Default is 0 (enabled).
constexpr const char* kDisableTrtFlashAttention = "ORT_DISABLE_TRT_FLASH_ATTENTION";

// Environment variable to enable or disable cutlass memory efficient attention. Default is 0 (enabled).
constexpr const char* kDisableMemoryEfficientAttention = "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION";

// Environment variable to enable or disable flash attention. Default is 0 (enabled).
constexpr const char* kDisableFlashAttention = "ORT_DISABLE_FLASH_ATTENTION";

// Minimum sequence length to perfer memory efficient attention when data type is float32
constexpr const char* kMinSeqLenForEfficientAttentionFp32 = "ORT_MIN_SEQ_LEN_EFFICIENT_ATTENTION_FP32";

// Default value for minimum sequence length to enable memory efficient attention in FP32.
constexpr int kDefaultMinSeqLenForEfficientAttentionFp32 = 256;

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
