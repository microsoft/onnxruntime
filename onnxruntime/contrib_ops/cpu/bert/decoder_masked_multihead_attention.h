// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class DecoderMaskedMultiHeadAttention final : public OpKernel, public AttentionCPUBase {
 public:
  DecoderMaskedMultiHeadAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
  bool past_present_share_buffer_;
  bool output_qk_;
};

}  // namespace contrib
}  // namespace onnxruntime
