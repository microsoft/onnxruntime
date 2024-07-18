// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"

using namespace onnxruntime::contrib::attention;

namespace onnxruntime {
void AttentionKernelOptions::Initialize(int value, bool use_build_flag) {
  if (value > 0) {
    use_flash_attention_ = (value & static_cast<int>(AttentionBackend::FLASH_ATTENTION)) > 0;
    use_efficient_attention_ = (value & static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION)) > 0;
    use_trt_fused_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION)) > 0;
    use_cudnn_flash_attention_ = (value & static_cast<int>(AttentionBackend::CUDNN_FLASH_ATTENTION)) > 0;
    use_unfused_ = (value & static_cast<int>(AttentionBackend::MATH)) > 0;
    use_trt_flash_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION)) > 0;
    use_trt_cross_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CROSS_ATTENTION)) > 0;
    use_trt_causal_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CAUSAL_ATTENTION)) > 0;
  } else {
    use_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFlashAttention, false);
    use_efficient_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableMemoryEfficientAttention, false);
    use_trt_fused_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFusedSelfAttention, false);
    use_cudnn_flash_attention_ = ParseEnvironmentVariableWithDefault<bool>(kEnableCudnnFlashAttention, false);
    use_unfused_ = true;
    use_trt_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableTrtFlashAttention, false);
    use_trt_cross_attention_ = !ParseEnvironmentVariableWithDefault<bool>(kDisableFusedCrossAttention, false);
    use_trt_causal_attention_ = ParseEnvironmentVariableWithDefault<bool>(kEnableFusedCausalAttention, false);
  }

  enable_kernel_debug_info_ = ParseEnvironmentVariableWithDefault<bool>(kEnableAttentionKernelDebugInfo, false);

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
  std::call_once(this->initialize_once_flag_, [&]() {
    this->Initialize(sdpa_kernel, use_build_flag);
    if (this->enable_kernel_debug_info_) {
      this->Print();
    }
  });
}

void AttentionKernelOptions::Print() const {
  std::stringstream sstream;
  sstream << "AttentionKernelOptions:";
  sstream << " FLASH_ATTENTION=" << int(use_flash_attention_);
  sstream << " EFFICIENT_ATTENTION=" << int(use_efficient_attention_);
  sstream << " TRT_FUSED_ATTENTION=" << int(use_trt_fused_attention_);
  sstream << " CUDNN_FLASH_ATTENTION=" << int(use_cudnn_flash_attention_);
  sstream << " TRT_FLASH_ATTENTION=" << int(use_trt_flash_attention_);
  sstream << " TRT_CROSS_ATTENTION=" << int(use_trt_cross_attention_);
  sstream << " TRT_CAUSAL_ATTENTION=" << int(use_trt_causal_attention_);
  sstream << " MATH=" << int(use_unfused_);

  if (!use_unfused_) {
    sstream << std::endl
            << "Warning: Unfused kernel cannot be disabled right now. MATH=0 is ignored.";
  }

  // Output text in Cyan color to make it easier to spot
  std::cout << "\x1B[36m" << sstream.str() << "\x1B[0m" << std::endl;
}

// Classify the kernel used in TRT fused runner.
void AttentionKernelDebugInfo::SetTrtFusedKernel(bool causal, bool enable_trt_flash_attention, int sequence_length) {
  if (causal) {
    use_trt_causal_attention = true;
  } else if (enable_trt_flash_attention && sequence_length >= contrib::cuda::kMinSequenceLengthFlashAttention) {
    use_trt_flash_attention = true;
  } else {
    use_trt_fused_attention = true;
  }
}

void AttentionKernelDebugInfo::Print(const char* operator_name,
                                     const std::string& node_name,
                                     bool is_float16,
                                     bool is_bfloat16) const {
  std::stringstream sstream;
  sstream << "Operator=" << operator_name;

  if (node_name.length() > 0) {
    sstream << " Node=" << node_name;
  }

  if (is_bfloat16) {
    sstream << " DataType=bf16";
  } else if (is_float16) {
    sstream << " DataType=fp16";
  } else {
    sstream << " DataType=fp32";
  }

  if (use_flash_attention.has_value() && use_flash_attention.value()) {
    sstream << " FLASH_ATTENTION=" << int(use_flash_attention.value());
  }

  if (use_efficient_attention.has_value() && use_efficient_attention.value()) {
    sstream << " EFFICIENT_ATTENTION=" << int(use_efficient_attention.value());
  }

  if (use_trt_fused_attention.has_value() && use_trt_fused_attention.value()) {
    sstream << " TRT_FUSED_ATTENTION=" << int(use_trt_fused_attention.value());
  }

  if (use_cudnn_flash_attention.has_value() && use_cudnn_flash_attention.value()) {
    sstream << " CUDNN_FLASH_ATTENTION=" << int(use_cudnn_flash_attention.value());
  }

  if (use_trt_flash_attention.has_value() && use_trt_flash_attention.value()) {
    sstream << " TRT_FLASH_ATTENTION=" << int(use_trt_flash_attention.value());
  }

  if (use_trt_cross_attention.has_value() && use_trt_cross_attention.value()) {
    sstream << " TRT_CROSS_ATTENTION=" << int(use_trt_cross_attention.value());
  }

  if (use_trt_causal_attention.has_value() && use_trt_causal_attention.value()) {
    sstream << " TRT_CAUSAL_ATTENTION=" << int(use_trt_causal_attention.value());
  }

  bool use_fused = (use_flash_attention.has_value() && use_flash_attention.value()) ||
                   (use_efficient_attention.has_value() && use_efficient_attention.value()) ||
                   (use_trt_fused_attention.has_value() && use_trt_fused_attention.value()) ||
                   (use_cudnn_flash_attention.has_value() && use_cudnn_flash_attention.value()) ||
                   (use_trt_flash_attention.has_value() && use_trt_flash_attention.value()) ||
                   (use_trt_cross_attention.has_value() && use_trt_cross_attention.value()) ||
                   (use_trt_causal_attention.has_value() && use_trt_causal_attention.value());

  // Fall back to unfused when no fused kernel is enabled.
  if (!use_fused) {
    sstream << " MATH=1";
  }

  // Output text in Cyan color to make it easier to spot.
  std::cout << "\x1B[36m" << sstream.str() << "\x1B[0m" << std::endl;
}

}  // namespace onnxruntime
