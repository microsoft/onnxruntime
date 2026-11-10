// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime::contrib::webgpu {

class GatedDeltaNetStateReplay final : public onnxruntime::webgpu::WebGpuKernel {
 public:
  explicit GatedDeltaNetStateReplay(const OpKernelInfo& info) : WebGpuKernel(info) {}

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;
};

}  // namespace onnxruntime::contrib::webgpu
