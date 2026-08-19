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
      const Tensor* past_key, const Tensor* past_value,
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

  // cuDNN SDPA decode tier (Phase 1: opset-24 external KV cache).
  // Reads the full `present` KV cache (K/V) and a per-batch valid length derived from
  // nonpad_kv_seqlen; produces GQA-class decode latency without any host-side valid-length
  // readback (CUDA-graph safe). Only reached for the narrowly gated decode case: external
  // cache (nonpad_kv_seqlen != nullptr, past_key == nullptr), q_sequence_length == 1,
  // no attn_mask / output_qk / softcap, fp16/bf16 (is_causal is not gated — for s_q==1 cuDNN
  // drops causal masking, so both is_causal values reduce to the same padding-only frontier).
  // See the cascade in ComputeInternal.
  Status RunCudnnSdpaAttention(
      OpKernelContext* context,
      const Tensor* Q, const Tensor* K, const Tensor* V,
      const Tensor* nonpad_kv_seqlen,
      Tensor* Y, Tensor* present_key, Tensor* present_value,
      const attention_helper::AttentionParameters& parameters) const;

  // Unified unfused fallback. Handles:
  //   - GQA (q_num_heads != kv_num_heads) without K/V head replication.
  //   - fp16/bf16 with large head_size (FP32 QK accumulation, fixes #28195).
  //   - past_key+past_value, attn_mask (bool/float), nonpad_kv_seqlen.
  //   - output_qk (kQK mode: scale * Q @ K^T, before softcap/mask/softmax).
  Status RunUnfusedAttention(
      OpKernelContext* context,
      const Tensor* Q, const Tensor* K, const Tensor* V,
      const Tensor* attn_mask, const Tensor* past_key, const Tensor* past_value,
      const Tensor* nonpad_kv_seqlen,
      Tensor* Y, Tensor* present_key, Tensor* present_value,
      Tensor* output_qk,
      const attention_helper::AttentionParameters& parameters) const;

  Status ConvertAttnMaskToBias(
      OpKernelContext* context,
      const Tensor* attn_mask,
      cudaStream_t cuda_stream,
      int max_threads_per_block,
      IAllocatorUniquePtr<void>& converted_mask_buffer,
      const void*& attn_bias_data,
      bool& broadcast_bias_dim_0,
      bool& broadcast_bias_dim_1) const;

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
  // cuDNN SDPA (cudnn_frontend) tier. NOTE: these use enable_/auto_enable_ polarity (not the
  // disable_ polarity of the two members above) deliberately, to mirror GroupQueryAttention's
  // cuDNN members verbatim (group_query_attention.h) — the cuDNN option defaults OFF, unlike
  // Flash/MEA which default ON, so an enable_ flag is the natural fit and keeps the two ops'
  // cuDNN gating identical. Reuses the shared cuDNN option surfaced via
  // AttentionKernelOptions::UseCudnnFlashAttention() / AllowCudnnFlashAttentionAuto() (no
  // Attention-specific option key). enable_ is set when the option is on (fp16/bf16 only);
  // auto_enable_ additionally auto-prefers cuDNN on SM>=90 unless the user pinned kernels or
  // disabled it.
  bool enable_cudnn_flash_attention_;
  bool auto_enable_cudnn_flash_attention_;
};

}  // namespace cuda
}  // namespace onnxruntime
