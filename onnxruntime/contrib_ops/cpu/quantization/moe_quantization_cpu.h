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

  int64_t expert_weight_bits_;
};

}  // namespace contrib
}  // namespace onnxruntime
