// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <mutex>
#include <optional>
#include <string>

namespace onnxruntime {
struct AttentionKernelDebugInfo {
  std::optional<bool> use_flash_attention = std::nullopt;
  std::optional<bool> use_lean_attention = std::nullopt;
  std::optional<bool> use_efficient_attention = std::nullopt;
  std::optional<bool> use_trt_fused_attention = std::nullopt;
  std::optional<bool> use_cudnn_flash_attention = std::nullopt;
  std::optional<bool> use_trt_flash_attention = std::nullopt;
  std::optional<bool> use_trt_cross_attention = std::nullopt;
  std::optional<bool> use_trt_causal_attention = std::nullopt;
  void SetTrtFusedKernel(bool causal, bool enable_trt_flash_attention, int sequence_length);
  void Print(const char* operator_name, const std::string& node_name, bool is_float16, bool is_bfloat16) const;
};

class AttentionKernelOptions {
 public:
  void InitializeOnce(int sdpa_kernel, bool use_build_flag, bool check_cudnn_version = false);

  bool UseFlashAttention() const { return use_flash_attention_; }
  bool UseLeanAttention() const { return use_lean_attention_; }
  bool UseEfficientAttention() const { return use_efficient_attention_; }
  bool UseTrtFusedAttention() const { return use_trt_fused_attention_; }
  bool UseCudnnFlashAttention() const { return use_cudnn_flash_attention_; }
  bool UseUnfusedAttention() const { return use_unfused_; }
  bool UseTrtFlashAttention() const { return use_trt_flash_attention_; }
  bool UseTrtCrossAttention() const { return use_trt_cross_attention_; }
  bool UseTrtCausalAttention() const { return use_trt_causal_attention_; }

  bool AllowDebugInfo() const { return enable_kernel_debug_info_; }

  int MinSeqLenForFlashAttentionPackedQkv() const { return min_seq_len_for_flash_attention_packed_qkv_; }
  int MinSeqLenForEfficientAttentionFp32() const { return min_seq_len_for_efficient_attention_fp32_; }

 protected:
  void Print() const;

  void Initialize(int value, bool use_build_flag, bool check_cudnn_version);

 private:
  bool use_flash_attention_{true};
  bool use_lean_attention_{false};
  bool use_efficient_attention_{true};
  bool use_trt_fused_attention_{true};
  bool use_cudnn_flash_attention_{false};
  bool use_unfused_{true};

  bool use_trt_flash_attention_{true};

  bool use_trt_cross_attention_{true};

  // Causal attention is disabled by default in #14732.
  bool use_trt_causal_attention_{false};

  bool enable_kernel_debug_info_{false};

  int min_seq_len_for_flash_attention_packed_qkv_{0};

  int min_seq_len_for_efficient_attention_fp32_{0};

  std::once_flag initialize_once_flag_;
};

}  // namespace onnxruntime
