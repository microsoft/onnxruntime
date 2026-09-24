// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/bert/linear_attention_gates_common.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// decay = decay_scale * Softplus(a + dt_bias), beta = Sigmoid(b).
template <typename T>
class LinearAttentionGate final : public OpKernel {
 public:
  explicit LinearAttentionGate(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

// Y = X * rsqrt(mean(X^2) + epsilon) * scale * gate_activation(gate).
template <typename T>
class GatedRMSNorm final : public OpKernel {
 public:
  explicit GatedRMSNorm(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  GatedRMSNormActivation activation_;
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
