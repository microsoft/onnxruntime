// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class SoftmaxDropoutGrad final : public CudaKernel {
 public:
  SoftmaxDropoutGrad(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace onnxruntime
