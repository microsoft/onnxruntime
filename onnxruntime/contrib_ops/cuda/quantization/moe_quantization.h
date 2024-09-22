// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/ft_moe/moe_kernel.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QMoE final : public CudaKernel, public MoEBase {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  template <typename CudaWeightT>
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
                          const Tensor* fc3_scales_optional,
                          const cudaDeviceProp& device_prop) const;

  int64_t expert_weight_bits_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
