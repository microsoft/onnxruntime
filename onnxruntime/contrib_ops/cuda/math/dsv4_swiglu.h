// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// The clamped SwiGLU DeepSeek-V4 uses in its shared expert, on a gate/up pair that has
// already been split into two tensors.
template <typename T>
class DSV4SwiGLU final : public CudaKernel {
 public:
  explicit DSV4SwiGLU(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float limit_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
