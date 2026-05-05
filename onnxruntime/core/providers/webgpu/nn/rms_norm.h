// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class RMSNorm final : public WebGpuKernel {
 public:
  RMSNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<float>("epsilon", &epsilon_, 1e-05f);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  float epsilon_;
};

}  // namespace webgpu
}  // namespace onnxruntime
