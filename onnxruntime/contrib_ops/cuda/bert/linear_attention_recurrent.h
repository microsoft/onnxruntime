// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

enum class LinearAttentionUpdateRule {
  kLinear,       // S_t = S_{t-1} + k_t ⊗ v_t
  kGated,        // S_t = exp(g_t) · S_{t-1} + k_t ⊗ v_t
  kDelta,        // S_t = S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}^T k_t)
  kGatedDelta,   // S_t = exp(g_t) · S_{t-1} + β_t · k_t ⊗ (v_t − exp(g_t) · S_{t-1}^T k_t)
};

template <typename T>
class LinearAttentionRecurrent final : public CudaKernel {
 public:
  LinearAttentionRecurrent(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  float scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
