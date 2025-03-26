// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/common/common.h"
#include "core/providers/cpu/controlflow/if.h"

namespace onnxruntime {
namespace webgpu {

// Use the CPU implementation for the logic
class If final : public onnxruntime::If, public WebGpuKernel {
 public:
  If(const OpKernelInfo& info) : onnxruntime::If(info), WebGpuKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace webgpu
}  // namespace onnxruntime