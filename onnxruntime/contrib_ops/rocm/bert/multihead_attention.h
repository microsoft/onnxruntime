// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/rocm/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class MultiHeadAttention final : public RocmKernel {
 public:
  MultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  AttentionType attn_type_;
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool past_present_share_buffer_{false};

  // type-erased GemmSoftmaxGemmPermuteTunableOp<HipT>, the reason for this is:
  //   1. We don't want to include the cuh file where GemmSoftmaxGemmPermuteTunableOp<HipT> is defined.
  //   2. We don't want to construct the object repeatly (which is expansive) during Compute.
  std::shared_ptr<void> tunable_op_;
};

template <typename T>
class DecoderMaskedMultiHeadAttention final : public RocmKernel {
 public:
  DecoderMaskedMultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  AttentionType mha_type;
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
