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

  // Batched CPU MoE implementation similar to CUDA approach
  Status MoEImplBatched(const Tensor* input, const Tensor* router_probs,
                        const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias,
                        const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias,
                        Tensor* output, int64_t num_rows, int64_t hidden_size, int64_t num_experts,
                        int64_t inter_size, int64_t fc1_inter_size, bool legacy_shape) const;

  // Batched expert processing using MLAS with row-major layout
  Status ProcessExpertBatch(const T* input_batch,
                            const T* fc1_weights, const T* fc1_bias,
                            const T* fc2_weights, const T* fc2_bias,
                            T* fc1_output_batch, T* final_output_batch,
                            int64_t batch_size, int64_t hidden_size, int64_t inter_size,
                            int64_t fc1_inter_size, bool legacy_shape) const;

  Status ProcessExpert(const T* input_data,
                       const T* fc1_weights,
                       const T* fc1_bias,
                       const T* fc2_weights,
                       const T* fc2_bias,
                       T* output_data,
                       int64_t hidden_size,
                       int64_t inter_size,
                       int64_t fc1_inter_size,
                       bool legacy_shape) const;

  void ApplyActivationInPlace(T* data, int64_t size, bool is_swiglu_format = false) const;
  void ApplySwiGLUActivationTyped(const T* input, T* output, int64_t size) const;

  Status ProcessGemm(const T* A, const T* B, T* C,
                     int64_t M, int64_t K, int64_t N, bool legacy_shape, bool is_fc1) const;
};

}  // namespace contrib
}  // namespace onnxruntime
