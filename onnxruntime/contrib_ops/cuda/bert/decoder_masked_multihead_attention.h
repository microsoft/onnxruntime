// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T1, typename T2>
class DecoderMaskedMultiHeadAttention final : public CudaKernel {
 public:
  DecoderMaskedMultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool past_present_share_buffer_;
};

template <typename T1, typename T2>
class DecoderMaskedMultiHeadAttentionQuantKV final : public CudaKernel {
 public:
  DecoderMaskedMultiHeadAttentionQuantKV(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool past_present_share_buffer_;
  int quant_kv_block_size_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
