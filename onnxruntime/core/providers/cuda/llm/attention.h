// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Attention final : public CudaKernel {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status RunFlashAttention(
      OpKernelContext* context,
      const Tensor* Q, const Tensor* K, const Tensor* V,
      const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
      const Tensor* nonpad_kv_seqlen,
      Tensor* Y, Tensor* present_key, Tensor* present_value,
      const attention_helper::AttentionParameters& parameters) const;

  Status RunMemoryEfficientAttention(
      OpKernelContext* context,
      const Tensor* Q, const Tensor* K, const Tensor* V,
      const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
      const Tensor* nonpad_kv_seqlen,
      Tensor* Y, Tensor* present_key, Tensor* present_value,
      const attention_helper::AttentionParameters& parameters) const;

  Status RunUnfusedAttention(
      OpKernelContext* context,
      const Tensor* Q, const Tensor* K, const Tensor* V,
      const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
      const Tensor* nonpad_kv_seqlen,
      Tensor* Y, Tensor* present_key, Tensor* present_value,
      Tensor* output_qk,
      const attention_helper::AttentionParameters& parameters) const;

 protected:
  bool is_causal_;
  int kv_num_heads_;
  int q_num_heads_;
  attention_helper::QKMatMulOutputMode qk_matmul_output_mode_;
  float scale_;
  float softcap_;
  int softmax_precision_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
};

}  // namespace cuda
}  // namespace onnxruntime
