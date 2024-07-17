// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::contrib::attention;

namespace onnxruntime {
void AttentionKernelOptions::Initialize(int value, bool use_build_flag) {
  if (value > 0) {
    use_flash_attention_ = (value & static_cast<int>(AttentionBackend::FLASH_ATTENTION)) > 0;
    use_efficient_attention_ = (value & static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION)) > 0;
    use_trt_fused_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION)) > 0;
    use_unfused_ = (value & static_cast<int>(AttentionBackend::MATH)) > 0;
    use_trt_flash_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION)) > 0;
    use_trt_cross_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CROSS_ATTENTION)) > 0;
    use_trt_causal_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CAUSAL_ATTENTION)) > 0;
  } else {
    use_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFlashAttention, false);
    use_efficient_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableMemoryEfficientAttention, false);
    use_trt_fused_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFusedSelfAttention, false);
    use_unfused_ = true;
    use_trt_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableTrtFlashAttention, false);
    use_trt_cross_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFusedCrossAttention, false);
    use_trt_causal_attention_ = ParseEnvironmentVariableWithDefault<bool>(kEnableFusedCausalAttention, false);
  }

  // When value is positive, we use 0 as default minimum sequence lengths to align with common usage in testing.
  min_seq_len_for_flash_attention_packed_qkv_ = ParseEnvironmentVariableWithDefault<int>(
      kMinSeqLenForFlashAttentionPackedQKV,
      value > 0 ? 0 : kDefaultMinSeqLenForFlashAttentionPackedQKV);

  min_seq_len_for_efficient_attention_fp32_ = ParseEnvironmentVariableWithDefault<int>(
      kMinSeqLenForEfficientAttentionFp32,
      value > 0 ? 0 : kDefaultMinSeqLenForEfficientAttentionFp32);

  if (use_build_flag) {
    // Some kernels can be disabled at build time. If they are disabled, we should not use them.
#ifndef USE_FLASH_ATTENTION
    use_flash_attention_ = false;
#endif

#ifndef USE_MEMORY_EFFICIENT_ATTENTION
    use_efficient_attention_ = false;
#endif
  }
}

void AttentionKernelOptions::InitializeOnce(
    int sdpa_kernel, bool use_build_flag) {
  std::call_once(this->initialize_once_flag_, [&]() { this->Initialize(sdpa_kernel, use_build_flag); });
}

}  // namespace onnxruntime
