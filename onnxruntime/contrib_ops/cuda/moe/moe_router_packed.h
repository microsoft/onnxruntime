// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class MoERouterPacked final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MoERouterPacked(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoERouterPacked);

 private:
  int64_t k_{0};
  float router_scale_{1.0f};
};

}  // namespace onnxruntime::contrib::cuda
