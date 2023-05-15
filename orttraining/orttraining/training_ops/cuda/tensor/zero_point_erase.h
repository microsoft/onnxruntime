// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

class ZeroPointErase final : public CudaKernel {
 public:
  ZeroPointErase(const OpKernelInfo& info) : CudaKernel(info) {
    default_zero_point_value_ = info.GetAttrOrDefault<float>("zero_point", 0.f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float default_zero_point_value_ = 0.0f;
};

}  // namespace cuda
}  // namespace onnxruntime
