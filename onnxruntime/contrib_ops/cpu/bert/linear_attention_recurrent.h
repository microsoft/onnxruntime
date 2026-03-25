// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// Mirrors LinearAttentionUpdateRule from the CUDA header —
// kept in the CPU namespace so both can coexist in the same build.
enum class LinearAttentionUpdateRule {
  kLinear,      // S_t = S_{t-1} + k_t ⊗ v_t
  kGated,       // S_t = exp(g_t) · S_{t-1} + k_t ⊗ v_t
  kDelta,       // S_t = S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}^T k_t)
  kGatedDelta,  // S_t = exp(g_t)·S_{t-1} + β_t·k_t ⊗ (v_t − exp(g_t)·S_{t-1}^T k_t)
};

template <typename T>
class LinearAttentionRecurrent final : public OpKernel {
 public:
  explicit LinearAttentionRecurrent(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  float scale_;

  // Compute one (batch, head) recurrent step entirely in float32.
  // state is updated in-place; output receives the query readout.
  void ComputeSingleHead(
      const float* q,      // [d_k]
      const float* k,      // [d_k]
      const float* v,      // [d_v]
      float* state,        // [d_k * d_v], updated in-place
      const float* decay,  // [d_k] — already exp(·), or nullptr for linear/delta
      float beta_val,      // scalar beta, 0 for linear/gated
      float* output,       // [d_v]
      int d_k, int d_v, float scale) const;
};

}  // namespace contrib
}  // namespace onnxruntime
