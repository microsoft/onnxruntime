// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MultiHeadAttention final : public CudaKernel {
 public:
  MultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool disable_fused_self_attention_;
  bool enable_trt_flash_attention_;
  bool disable_fused_cross_attention_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  int min_seq_len_for_flash_attention_packed_qkv_;
  mutable std::unique_ptr<MHARunner> fused_fp16_runner_;
  mutable std::once_flag fused_fp16_runner_created_;
  mutable const FusedMultiHeadCrossAttentionKernel* fused_fp16_cross_attention_kernel_;
  mutable CumulatedSequenceLengthCache cumulated_sequence_length_q_cache_;
  mutable CumulatedSequenceLengthCache cumulated_sequence_length_kv_cache_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
