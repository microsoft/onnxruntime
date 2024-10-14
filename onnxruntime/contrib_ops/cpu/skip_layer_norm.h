// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

template <typename T, bool simplified>
class SkipLayerNorm final : public OpKernel {
 public:
  SkipLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  float epsilon_;
  mutable IAllocatorUniquePtr<float> skip_fp32_;
  mutable IAllocatorUniquePtr<float> gamma_fp32_;
  mutable IAllocatorUniquePtr<float> beta_fp32_;
  mutable IAllocatorUniquePtr<float> bias_fp32_;
};

}  // namespace contrib
}  // namespace onnxruntime
