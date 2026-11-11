// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/linear_attention_gates_helper.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

using namespace onnxruntime::contrib::linear_attention_gates_helper;

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

template <typename T>
class MoEGroupRouter final : public OpKernel {
 public:
  explicit MoEGroupRouter(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t top_k_;
  int64_t n_group_;
  int64_t topk_group_;
  int64_t group_topk_;
  float router_sentinel_;
};

}  // namespace contrib
}  // namespace onnxruntime
