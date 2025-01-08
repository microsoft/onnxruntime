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
  int64_t prepacked_skip_fp32_size_;
  IAllocatorUniquePtr<float> prepacked_skip_fp32_data_;
  IAllocatorUniquePtr<float> prepacked_gamma_fp32_data_;
  IAllocatorUniquePtr<float> prepacked_beta_fp32_data_;
  IAllocatorUniquePtr<float> prepacked_bias_fp32_data_;
};

}  // namespace contrib
}  // namespace onnxruntime
