// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// First projection of a MoE layer with a shared expert: one GEMM whose columns are split into the router logits and
// the gate/up halves of the shared expert, followed by the gated activation (MatMul + Split + SwiGLU).
template <typename T>
class RouterGateUpProj final : public CudaKernel {
 public:
  explicit RouterGateUpProj(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t router_size_;
  float limit_;
  float alpha_;
  float beta_;
  bool weight_is_constant_{false};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
