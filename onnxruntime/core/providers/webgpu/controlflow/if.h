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
class If final : public WebGpuKernel {
 public:
  If(const OpKernelInfo& info) : WebGpuKernel(info), cpu_if_(info) {}

  Status ComputeInternal(ComputeContext& context) const override {
    return cpu_if_.Compute(context.KernelContext());
  }

 private:
  onnxruntime::If cpu_if_;
};

}  // namespace webgpu
}  // namespace onnxruntime