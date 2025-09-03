// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

namespace onnxruntime {
namespace contrib {

class MoE final : public OpKernel, public MoEBaseCPU {
 public:
  explicit MoE(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  Status MoEImpl(OpKernelContext* context,
                 const Tensor* input,
                 const Tensor* router_probs,
                 const Tensor* fc1_experts_weights,
                 const Tensor* fc1_experts_bias,
                 const Tensor* fc2_experts_weights,
                 const Tensor* fc2_experts_bias,
                 const Tensor* fc3_experts_weights,
                 const Tensor* fc3_experts_bias,
                 Tensor* output) const;

  void ProcessExpert(const float* input_data,
                     const float* fc1_weights,
                     const float* fc1_bias,
                     const float* fc2_weights,
                     const float* fc2_bias,
                     float* output_data,
                     int64_t hidden_size,
                     int64_t inter_size,
                     int64_t fc1_inter_size,
                     bool legacy_shape) const;

  void ApplyActivationInPlace(float* data, int64_t size, bool is_swiglu_format = false) const;
};

}  // namespace contrib
}  // namespace onnxruntime
