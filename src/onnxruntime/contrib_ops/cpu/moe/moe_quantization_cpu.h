// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

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
  int64_t expert_weight_bits_;
  int64_t block_size_;
};

}  // namespace contrib
}  // namespace onnxruntime
