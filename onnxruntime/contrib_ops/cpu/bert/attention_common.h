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
  AttentionKernel_LeanAttention,
  AttentionKernel_DecoderAttention,
  AttentionKernel_Default
};

enum class QKOutputType : int {
  NO_OUTPUT = 0,
  BEFORE_SOFTMAX = 1,
  AFTER_SOFTMAX = 2
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

  // The following TRT kernels might be deprecated in the future.
  TRT_FLASH_ATTENTION = 32,
  TRT_CROSS_ATTENTION = 64,
  TRT_CAUSAL_ATTENTION = 128,

  // Experimental kernels
  LEAN_ATTENTION = 256,
  DECODER_ATTENTION = 512,  // FasterTransformer's decoder masked multihead attention
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

// Environment variable to enable or disable lean attention. Default is 0 (disabled).
constexpr const char* kEnableLeanAttention = "ORT_ENABLE_LEAN_ATTENTION";

// Environment variable to enable or disable FasterTransformer's decoder masked multi-head attention. Default is 0 (enabled).
constexpr const char* kDisableDecoderAttention = "ORT_DISABLE_DECODER_ATTENTION";

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
