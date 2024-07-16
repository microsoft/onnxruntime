// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class AttentionBackend : int {
  FLASH_ATTENTION = 1,
  EFFICIENT_ATTENTION = 2,
  TRT_FUSED_ATTENTION = 4,
  MATH = 8,  // unfused

  // The following kernels might be deprected in the future.
  TRT_FLASH_ATTENTION = 16,
  TRT_CROSS_ATTENTION = 32,
  TRT_CAUSAL_ATTENTION = 64,
};

class AttentionKernelOptions {
 public:
  static const AttentionKernelOptions* GetInstance(int sdpa_kernel, bool force_init);

  bool UseFlashAttention() const { return use_flash_attention_; }
  bool UseEfficientAttention() const { return use_efficient_attention_; }
  bool UseTrtFusedAttention() const { return use_trt_fused_attention_; }
  bool UseUnfusedAttention() const { return use_unfused_; }
  bool UseTrtFlashAttention() const { return use_trt_flash_attention_; }
  bool UseTrtCrossAttention() const { return use_trt_cross_attention_; }
  bool UseTrtCausalAttention() const { return use_trt_causal_attention_; }

  int MinSeqLenForFlashAttentionPackedQkv() const { return min_seq_len_for_flash_attention_packed_qkv_; }
  int MinSeqLenForEfficientAttentionFp32() const { return min_seq_len_for_efficient_attention_fp32_; }

 protected:
  void Initialize(int value);

 private:
  bool use_flash_attention_{true};
  bool use_efficient_attention_{true};
  bool use_trt_fused_attention_{true};
  bool use_unfused_{true};
  bool use_trt_flash_attention_{true};
  bool use_trt_cross_attention_{true};

  // Causal attention is disabled by default in #14732.
  bool use_trt_causal_attention_{false};

  int min_seq_len_for_flash_attention_packed_qkv_{0};

  int min_seq_len_for_efficient_attention_fp32_{0};

  bool initialized_{false};
  static AttentionKernelOptions instance;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
