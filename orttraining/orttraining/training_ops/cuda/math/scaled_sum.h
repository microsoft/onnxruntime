// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class ScaledSum final : public CudaKernel {
 public:
  ScaledSum(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("scale_0", &scale_0_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("scale_1", &scale_1_).IsOK());
    info.GetAttrOrDefault<float>("scale_2", &scale_2_, 1.f);
  };
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float scale_0_;
  float scale_1_;
  float scale_2_;
};

}  // namespace cuda
}  // namespace onnxruntime
