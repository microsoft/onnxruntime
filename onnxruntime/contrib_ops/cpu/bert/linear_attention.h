// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {

// Update rule types for linear attention
enum class LinearAttentionUpdateRule {
  kLinear,       // S_t = S_{t-1} + k_t outer v_t
  kGated,        // S_t = exp(g_t) * S_{t-1} + k_t outer v_t
  kDelta,        // S_t = S_{t-1} + beta_t * k_t outer (v_t - S_{t-1}^T k_t)
  kGatedDelta,   // S_t = exp(g_t) * S_{t-1} + beta_t * k_t outer (v_t - exp(g_t) * S_{t-1}^T k_t)
};

template <typename T>
class LinearAttention final : public OpKernel {
 public:
  LinearAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  float scale_;
};

}  // namespace contrib
}  // namespace onnxruntime
