// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {
ComputeContext::ComputeContext(OpKernelContext& kernel_context)
    : webgpu_context_{WebGpuContextFactory::GetContext(kernel_context.GetDeviceId())},
      kernel_context_{kernel_context} {
}

const wgpu::AdapterInfo& ComputeContext::AdapterInfo() const {
  return webgpu_context_.AdapterInfo();
}

const wgpu::Limits& ComputeContext::DeviceLimits() const {
  return webgpu_context_.DeviceLimits();
}

int ComputeContext::InputCount() const {
  return kernel_context_.InputCount();
}

int ComputeContext::OutputCount() const {
  return kernel_context_.OutputCount();
}

Status ComputeContext::RunProgram(const ProgramBase& program) {
  return webgpu_context_.Run(*this, program);
}

}  // namespace webgpu
}  // namespace onnxruntime
