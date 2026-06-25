// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas_q4.h"
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
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers_V2(std::vector<BufferUniquePtr>& prepacked_buffers,
                                      gsl::span<const size_t> prepacked_buffer_sizes,
                                      int input_idx,
                                      /*out*/ bool& used_shared_buffers) override;

  void ApplyActivationVectorized(float* data, int64_t size) const;

  int64_t expert_weight_bits_;
  int64_t block_size_;
  bool use_mlas_q4_gemm_{false};
  bool use_mlas_q4_gemm_overridden_{false};

  IAllocatorUniquePtr<void> packed_fc1_;
  IAllocatorUniquePtr<void> packed_fc2_;

  TensorShape fc1_shape_;
  TensorShape fc2_shape_;

  IAllocatorUniquePtr<void> packed_fc1_mlas_cache_;
  IAllocatorUniquePtr<void> packed_fc2_mlas_cache_;
};

}  // namespace contrib
}  // namespace onnxruntime
