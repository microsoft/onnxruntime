// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

/**
 * @brief QMoE is the templated CPU implementation of the Quantized Mixture of Experts operator.
 *
 * This kernel supports both float and MLFloat16 data types for activations, scales, and outputs.
 * It parallelizes expert computation using the ONNX Runtime thread pool and minimizes memory
 * usage through on-the-fly block dequantization of weights.
 *
 * @tparam T The data type for the kernel (float or MLFloat16).
 */
template <typename T>
class QMoECPU final : public OpKernel, public MoEBaseCPU {
 public:
  explicit QMoECPU(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 private:
  struct ComputeInputs {
    const Tensor* input;
    const Tensor* router_probs;
    const Tensor* fc1_experts_weights;
    const Tensor* fc1_scales;
    const Tensor* fc1_experts_bias;
    const Tensor* fc2_experts_weights;
    const Tensor* fc2_scales;
    const Tensor* fc2_experts_bias;
    const Tensor* fc3_experts_weights;
    const Tensor* fc3_scales;
    const Tensor* fc3_experts_bias;
    const Tensor* fc1_zero_points;
    const Tensor* fc2_zero_points;
    const Tensor* fc3_zero_points;
    const Tensor* router_weights;
  };

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  Status ComputeCommon(OpKernelContext* context, const ComputeInputs& inputs, const MoEParameters& moe_params) const;

  void ApplyActivationVectorized(float* data, int64_t size) const;

  int64_t expert_weight_bits_;
  int64_t block_size_;
  bool use_mlas_q4_gemm_{false};
  bool use_mlas_q4_gemm_overridden_{false};

  IAllocatorUniquePtr<void> packed_fc1_;
  IAllocatorUniquePtr<void> packed_fc2_;
  IAllocatorUniquePtr<void> packed_fc1_lut_cache_;
  IAllocatorUniquePtr<void> packed_fc2_lut_cache_;

  TensorShape fc1_shape_;
  TensorShape fc2_shape_;

  IAllocatorUniquePtr<void> packed_fc1_mlas_cache_;
  IAllocatorUniquePtr<void> packed_fc2_mlas_cache_;
};

}  // namespace contrib
}  // namespace onnxruntime
