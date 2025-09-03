// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MoE final : public OpKernel, public MoEBaseCPU {
 public:
  explicit MoE(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeMoE(const OpKernelContext* context,
                    const Tensor* input,
                    const Tensor* router_probs,
                    const Tensor* fc1_experts_weights,
                    const Tensor* fc1_experts_bias,
                    const Tensor* fc2_experts_weights,
                    const Tensor* fc2_experts_bias,
                    Tensor* output) const;

  Status ProcessExpertBatch(const T* input_tokens,
                            const int64_t* token_expert_ids,
                            const float* token_weights,
                            int64_t num_tokens,
                            int64_t expert_id,
                            const T* fc1_weights,
                            const T* fc1_bias,
                            const T* fc2_weights,
                            const T* fc2_bias,
                            T* output_buffer,
                            int64_t hidden_size,
                            int64_t inter_size) const;

  Status ComputeGEMM(const T* A, const T* B, T* C,
                     int64_t M, int64_t K, int64_t N,
                     bool transpose_B = false) const;

  void ApplyActivationVectorized(T* data, int64_t size) const;
  void ApplySwiGLUVectorized(const T* input, T* output, int64_t size) const;
};

}  // namespace contrib
}  // namespace onnxruntime
