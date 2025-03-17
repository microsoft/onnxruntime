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

void ComputeContext::PushErrorScope() {
  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    webgpu_context_.PushErrorScope();
  }
}

Status ComputeContext::PopErrorScope() {
  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    return webgpu_context_.PopErrorScope();
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
