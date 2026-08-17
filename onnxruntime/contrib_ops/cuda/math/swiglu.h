// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Clamped SwiGLU on a gate/up pair given either as two tensors or as the two halves of one
// `[.., 2 * inter]` projection.  Same alpha/beta/limit contract MoE and QMoE apply internally.
template <typename T>
class SwiGLU final : public CudaKernel {
 public:
  explicit SwiGLU(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float limit_;
  float alpha_;
  float beta_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
