// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class ScaledSum final : public CudaKernel {
 public:
  ScaledSum(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("scale_0", &scale0_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("scale_1", &scale1_).IsOK());
    float scale2_tmp;
    if (info.GetAttr<float>("scale_2", &scale2_tmp).IsOK()) {
      scale2_ = scale2_tmp;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float scale0_;
  float scale1_;
  std::optional<float> scale2_;
};

}  // namespace cuda
}  // namespace onnxruntime
