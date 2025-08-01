// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/moe/moe_base_cpu.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class QMoE final : public OpKernel, public MoEBaseCPU {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* ctx) const override;

 private:
  template <bool UseUInt4x2>
  Status PrepackAndDequantizeWeights(OpKernelContext* context,
                                     MoEParameters& moe_params,
                                     const Tensor* fc1_experts_weights,
                                     const Tensor* fc2_experts_weights,
                                     const Tensor* fc1_scales,
                                     const Tensor* fc2_scales,
                                     bool is_swiglu);

  template <bool UseUInt4x2, typename T>
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

  // Prepacked dequantized weights stored for reuse
  std::vector<float> prepacked_fc1_weights_;
  std::vector<float> prepacked_fc2_weights_;

  // Cached parameters to detect changes requiring repack
  mutable int64_t cached_num_experts_{0};
  mutable int64_t cached_hidden_size_{0};
  mutable int64_t cached_inter_size_{0};
  mutable bool cached_is_swiglu_{false};
  mutable bool is_prepacked_{false};

  int64_t expert_weight_bits_;
};

}  // namespace contrib
}  // namespace onnxruntime
