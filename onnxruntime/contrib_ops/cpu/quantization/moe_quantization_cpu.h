// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/moe/moe_base_cpu.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class QMoE final : public OpKernel, public MoEBaseCPU {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* ctx) const override;

 private:
  template <bool UseUInt4x2>
  Status QuantizedMoEImpl(OpKernelContext* context,
                          MoEParameters& moe_params,
                          const Tensor* input,
                          const Tensor* router_probs,
                          const Tensor* fc1_experts_weights,
                          const Tensor* fc1_experts_bias_optional,
                          const Tensor* fc2_experts_weights,
                          const Tensor* fc2_experts_bias_optional,
                          const Tensor* fc3_experts_weights_optional,
                          const Tensor* fc3_experts_bias_optional,
                          const Tensor* fc1_scales,
                          const Tensor* fc2_scales,
                          const Tensor* fc3_scales_optional) const;

  Status DirectFP32MoEImpl(OpKernelContext* context,
                           MoEParameters& moe_params,
                           const Tensor* input,
                           const Tensor* router_probs,
                           const Tensor* fc1_experts_weights,
                           const Tensor* fc1_experts_bias_optional,
                           const Tensor* fc2_experts_weights,
                           const Tensor* fc2_experts_bias_optional,
                           const Tensor* fc3_experts_weights_optional,
                           const Tensor* fc3_experts_bias_optional) const;

  template <bool UseUInt4x2>
  Status PrepackAndDequantizeWeights(OpKernelContext* context,
                                     MoEParameters& moe_params,
                                     const Tensor* fc1_experts_weights,
                                     const Tensor* fc2_experts_weights,
                                     const Tensor* fc1_scales,
                                     const Tensor* fc2_scales,
                                     bool is_swiglu);

  int64_t expert_weight_bits_;

  int64_t block_size_;

  // Prepacked dequantized weights stored for reuse
  IAllocatorUniquePtr<float> prepacked_fc1_weights_;
  IAllocatorUniquePtr<float> prepacked_fc2_weights_;
  float* prepacked_fc1_weights_data_{nullptr};
  float* prepacked_fc2_weights_data_{nullptr};

  // Persistent allocator for weights
  AllocatorPtr weights_allocator_;

  // Cached parameters to detect changes requiring repack
  mutable int64_t cached_num_experts_{0};
  mutable int64_t cached_hidden_size_{0};
  mutable int64_t cached_inter_size_{0};
  mutable bool cached_is_swiglu_{false};
  mutable bool is_prepacked_{false};
};

}  // namespace contrib
}  // namespace onnxruntime
