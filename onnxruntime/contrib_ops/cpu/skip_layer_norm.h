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

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  float epsilon_;
  TensorShape prepacked_skip_shape_;
  TensorShape prepacked_gamma_shape_;
  TensorShape prepacked_beta_shape_;
  TensorShape prepacked_bias_shape_;
  bool has_prepacked_skip_;
  bool has_prepacked_gamma_;
  bool has_prepacked_beta_;
  bool has_prepacked_bias_;
  // f32 copies of narrow float initializers made in PrePack(). Owned by the kernel, or by the session when it
  // caches the pre-packed weights (see UseSharedPrePackedBuffers()).
  IAllocatorUniquePtr<void> prepacked_skip_fp32_data_;
  IAllocatorUniquePtr<void> prepacked_gamma_fp32_data_;
  IAllocatorUniquePtr<void> prepacked_beta_fp32_data_;
  IAllocatorUniquePtr<void> prepacked_bias_fp32_data_;
};

}  // namespace contrib
}  // namespace onnxruntime
