// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <gsl/gsl>

#include <algorithm>
#include <cctype>
#include <string>
#include "core/common/common.h"

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

// Enum to define quantization granularity.
enum class KVQuantizationType : int {
  NONE = 0,
  PER_TENSOR = 1,
  PER_CHANNEL = 2,
  // Per-token, per-group asymmetric (scale + zero-point). Used by the OSCAR 2-bit KV cache:
  // each KV row is split into groups of kv_quant_group_size channels, and every group gets
  // its own scale/zero computed dynamically at append time and stored alongside the codes.
  PER_GROUP = 3,
};

inline KVQuantizationType StringToKVQuantizationType(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  if (s == "NONE") {
    return KVQuantizationType::NONE;
  }
  if (s == "PER_TENSOR") {
    return KVQuantizationType::PER_TENSOR;
  }
  if (s == "PER_CHANNEL") {
    return KVQuantizationType::PER_CHANNEL;
  }
  if (s == "PER_GROUP") {
    return KVQuantizationType::PER_GROUP;
  }
  ORT_THROW("Invalid KV quantization type: '", s,
            "'. Valid values are: NONE, PER_TENSOR, PER_CHANNEL, PER_GROUP.");
}

// Logical element type of a KV cache. Members are named after the ONNX element type they denote.
// DEFAULT means "whatever the cache tensor's own element type is" and is the only value a model
// needs when that type is expressible in ONNX, i.e. for float16 / bfloat16 / int8 / float8e4m3fn;
// naming one of those explicitly is allowed but must agree with the tensor.
// The sub-byte members have no ONNX tensor type here: they are packed two per byte into a uint8
// cache, so the last cache dimension holds (head_size + 1) / 2 bytes and logical element 2*i
// occupies the low-order bits of byte i.
// Every member is a *signed*, zero-symmetric type. Quantization here has a scale but no zero point
// (there are no zero-point inputs), so an unsigned logical type such as uint4 or uint8 would need
// an implied offset of 2^(bits-1) that nothing in the contract can express. INT4 is still *stored*
// in an unsigned nibble biased by +8, but that is a storage encoding removed on read, not a
// quantization zero point. Unsigned types must arrive together with zero-point inputs.
enum class KVCacheDataType : int {
  DEFAULT = 0,
  FLOAT16 = 1,
  BFLOAT16 = 2,
  INT8 = 3,
  FLOAT8E4M3FN = 4,
  INT4 = 5,
  FLOAT4E2M1 = 6,
};

// True for the packed sub-byte members, which are stored in a uint8 cache.
inline bool IsSubByteKVCacheDataType(KVCacheDataType t) {
  return t == KVCacheDataType::INT4 || t == KVCacheDataType::FLOAT4E2M1;
}

// True for the members that require a scale on read/write.
inline bool IsQuantizedKVCacheDataType(KVCacheDataType t) {
  return t != KVCacheDataType::DEFAULT && t != KVCacheDataType::FLOAT16 && t != KVCacheDataType::BFLOAT16;
}

inline const char* KVCacheDataTypeToString(KVCacheDataType t) {
  switch (t) {
    case KVCacheDataType::FLOAT16:
      return "float16";
    case KVCacheDataType::BFLOAT16:
      return "bfloat16";
    case KVCacheDataType::INT8:
      return "int8";
    case KVCacheDataType::FLOAT8E4M3FN:
      return "float8e4m3fn";
    case KVCacheDataType::INT4:
      return "int4";
    case KVCacheDataType::FLOAT4E2M1:
      return "float4e2m1";
    default:
      return "";
  }
}

inline KVCacheDataType StringToKVCacheDataType(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s.empty()) {
    return KVCacheDataType::DEFAULT;
  }
  if (s == "float16") {
    return KVCacheDataType::FLOAT16;
  }
  if (s == "bfloat16") {
    return KVCacheDataType::BFLOAT16;
  }
  if (s == "int8") {
    return KVCacheDataType::INT8;
  }
  if (s == "float8e4m3fn") {
    return KVCacheDataType::FLOAT8E4M3FN;
  }
  if (s == "int4") {
    return KVCacheDataType::INT4;
  }
  if (s == "float4e2m1") {
    return KVCacheDataType::FLOAT4E2M1;
  }
  ORT_THROW("Invalid KV cache data type: '", s,
            "'. Valid values are: '' (use the cache tensor's element type), float16, bfloat16, int8, "
            "float8e4m3fn, int4, float4e2m1. Unsigned types are excluded because quantization here is "
            "symmetric with no zero point.");
}

constexpr bool LAYOUT_BSNH = false;
constexpr bool LAYOUT_BNSH = true;

// Upper bound on the `state_window` attribute of LinearAttention and CausalConvWithState. The
// window only has to cover the tokens a multi-token predictor can propose, and the state tensors
// grow linearly with it, so cap it rather than let a model request an unbounded allocation.
constexpr int64_t kMaxStateWindow = 8;

namespace sparse_attention {
// Environment variable to enable or disable sparse attention v1 kernel. Default is 0 (enabled).
constexpr const char* kDisableSparseAttentionV1 = "ORT_DISABLE_SPARSE_ATTENTION_V1";

// Environment variable to disable device-side validation of CSR indices and key sequence lengths.
// Default is 0 (validation enabled). Set to 1 to skip the validation kernel launch and stream
// synchronization, which may improve latency when inputs are known to be well-formed.
// Usage: export ORT_DISABLE_SPARSE_ATTENTION_INPUT_VALIDATION=1
constexpr const char* kDisableInputValidation = "ORT_DISABLE_SPARSE_ATTENTION_INPUT_VALIDATION";
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

// Environment variable to enable or disable cuDNN flash attention.
constexpr const char* kEnableCudnnFlashAttention = "ORT_ENABLE_CUDNN_FLASH_ATTENTION";

// Environment variable to enable or disable TRT flash attention. Default is 0 (enabled).
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
