// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Sequence-length-independent (seq-free) route-eligibility predicates for the CUDA
// GroupQueryAttention operator. These factor the backend-selection math in
// GroupQueryAttention::ComputeInternal so the runtime and (in a later change) the
// Level-1 workspace estimator can share one definition of which attention backends a
// given node may dispatch to, rather than maintaining two copies that can drift.
//
// "Seq-free" means the predicate depends only on partition-time facts (geometry,
// device capability, quantization, feature flags) and NOT on the per-step sequence
// length or phase (prompt vs decode). The caller combines each seq-free core with the
// remaining seq-dependent gates and dispatch priorities:
//   - XQA:   phase gates (is_first_prompt / sequence_length == 1 / kv_sequence_length)
//            and the per-node shared-memory capability probe.
//   - cuDNN: onnxruntime::cudnn_sdpa::is_stable() and is_supported(..., seq_len_q,
//            seq_len_kv, ...), which are sequence-length dependent.
//   - Flash/MEA: the !use_xqa / !use_cudnn_sdpa / !use_flash_attention dispatch order.
//
// The functions are templated on the Q/KV element types, so this is a header-only
// module.

#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include "core/common/float16.h"
#include "core/common/float8.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#if USE_FLASH_ATTENTION
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#endif
#if USE_MEMORY_EFFICIENT_ATTENTION
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Seq-free inputs for the XQA decode-kernel eligibility core. qk_norm_ok and
// smooth_softmax_supported are the runtime's precomputed feature guards
// (!use_qk_norm || !is_inputs_quantized) and (!use_smooth_softmax || has_head_sink).
struct GQAXqaSeqFreeInputs {
  bool enable_xqa = false;
  bool is_unidirectional = false;
  bool has_attention_bias = false;
  int device_major = 0;
  int device_minor = 0;
  bool past_present_share_buffer = false;
  float softcap = 0.0f;
  bool qk_norm_ok = false;
  bool smooth_softmax_supported = false;
  bool is_inputs_quantized = false;
  int64_t head_size = 0;
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  KVQuantizationType k_quant_type = KVQuantizationType::NONE;
  KVQuantizationType v_quant_type = KVQuantizationType::NONE;
};

// Whether the XQA kernel is eligible ignoring phase gates and the shared-memory probe.
// U is the KV-cache element type; it selects which quantized XQA variant may run.
template <typename U>
inline bool IsGQAXqaEligibleSeqFree(const GQAXqaSeqFreeInputs& in) {
  if (!(in.enable_xqa &&
        in.is_unidirectional &&
        !in.has_attention_bias &&
        in.device_major >= 8 &&
        in.past_present_share_buffer &&
        in.softcap == 0.0f &&
        in.qk_norm_ok &&
        in.smooth_softmax_supported)) {
    return false;
  }

  constexpr bool is_int8 = std::is_same<U, int8_t>::value;
  const int64_t group_size = in.num_heads / in.kv_num_heads;
  // Sliding window (local_window_size > 0) is wired through to the quantized XQA kernels as well,
  // so the INT8/FP8 variants are not restricted to global attention. K and V may use different
  // scales: for PER_TENSOR the kernel folds k_scale into qkScale (applied to Q*K.T before softmax)
  // and v_scale into voScale (applied to the P*V accumulator). PER_CHANNEL scales cannot be scalars
  // inside the kernel, so ExtremeDecoding folds them into Q and into the attention output instead,
  // which is exact and costs two O(num_heads * head_size) passes -- far cheaper than dequantizing
  // the whole cache on every decode step.
  const auto is_supported_quant_type = [](KVQuantizationType t) {
    return t == KVQuantizationType::PER_TENSOR || t == KVQuantizationType::PER_CHANNEL;
  };

  const bool is_int8_quantized_supported =
      is_int8 &&
      (is_supported_quant_type(in.k_quant_type) &&
       is_supported_quant_type(in.v_quant_type) &&
       IsSupportedGQAXqaHeadSize(in.head_size) &&
       IsSupportedGQAXqaGroupSize(group_size, /*is_quantized=*/true));

#ifdef USE_FP8_KV_CACHE
  constexpr bool is_fp8 = std::is_same<U, Float8E4M3FN>::value;
  const bool is_fp8_quantized_supported =
      is_fp8 &&
      (is_supported_quant_type(in.k_quant_type) &&
       is_supported_quant_type(in.v_quant_type) &&
       IsSupportedGQAXqaHeadSize(in.head_size) &&
       IsSupportedGQAXqaGroupSize(group_size, /*is_quantized=*/true) &&
       (in.device_major >= 9 || (in.device_major == 8 && in.device_minor == 9)));  // FP8 requires SM89+ (Ada Lovelace)
#else
  constexpr bool is_fp8_quantized_supported = false;
#endif

  const bool is_non_quantized_supported =
      !in.is_inputs_quantized &&
      IsSupportedGQAXqaHeadSize(in.head_size) &&
      IsSupportedGQAXqaGroupSize(group_size, /*is_quantized=*/false);

  return is_non_quantized_supported || is_int8_quantized_supported || is_fp8_quantized_supported;
}

// Seq-free core of cuDNN SDPA eligibility. The runtime additionally requires the
// dispatch-priority gate (!use_xqa), onnxruntime::cudnn_sdpa::is_stable(), and the
// sequence-length-dependent onnxruntime::cudnn_sdpa::is_supported(...) probe.
// T/U are the Q and KV-cache element types (cuDNN requires a non-quantized cache, i.e.
// matching element types). cudnn_enabled is the runtime's combined enable gate
// (enable_cudnn_flash_attention_ || (auto_enable_cudnn_flash_attention_ && major >= 9)).
template <typename T, typename U>
inline bool IsGQACudnnSdpaCoreEligibleSeqFree(bool has_attention_bias,
                                              bool is_inputs_quantized,
                                              float softcap,
                                              bool use_smooth_softmax,
                                              bool has_head_sink,
                                              int local_window_size,
                                              bool past_kv_format_bnsh,
                                              bool cudnn_enabled) {
  return !has_attention_bias &&
         !is_inputs_quantized &&
         std::is_same<T, U>::value &&
         softcap == 0.0f &&
         !use_smooth_softmax &&
         !has_head_sink &&
         local_window_size == -1 &&
         past_kv_format_bnsh &&
         cudnn_enabled;
}

#if USE_FLASH_ATTENTION
// Whether FlashAttention is eligible ignoring the !use_xqa / !use_cudnn_sdpa dispatch
// priority. T is the Q element type (selects the compiled kernel set).
template <typename T>
inline bool IsGQAFlashEligibleSeqFree(const cudaDeviceProp& device_prop,
                                      bool has_attention_bias,  // flash_api.h has no bias parameter
                                      bool disable_flash_attention,
                                      int64_t head_size,
                                      int64_t num_heads,
                                      int64_t kv_num_heads) {
  return !has_attention_bias &&
         !disable_flash_attention &&
         onnxruntime::flash::is_supported<T>(device_prop,
                                             static_cast<size_t>(head_size),
                                             static_cast<size_t>(num_heads),
                                             static_cast<size_t>(kv_num_heads));
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
// Whether memory-efficient (CUTLASS FMHA) attention is eligible ignoring the
// !use_xqa / !use_cudnn_sdpa / !use_flash_attention dispatch priority. T is the Q
// element type. sm is the packed compute capability (major * 10 + minor).
template <typename T>
inline bool IsGQAMemoryEfficientEligibleSeqFree(int32_t sm,
                                                bool disable_memory_efficient_attention,
                                                bool is_inputs_quantized,
                                                bool has_attention_bias,
                                                int64_t head_size) {
  return !disable_memory_efficient_attention &&
         !is_inputs_quantized &&
         !has_attention_bias &&
         has_memory_efficient_attention(sm,
                                        std::is_same<T, MLFloat16>::value,
                                        std::is_same<T, BFloat16>::value,
                                        static_cast<int>(head_size),
                                        static_cast<int>(head_size));
}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
