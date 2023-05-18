// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

class ModeRestore final : public CudaKernel {
 public:
  ModeRestore(const OpKernelInfo& info) : CudaKernel(info) {
    mode_ = info.GetAttrOrDefault<float>("mode", 0.f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float mode_ = 0.0f;
};

}  // namespace cuda
}  // namespace onnxruntime
