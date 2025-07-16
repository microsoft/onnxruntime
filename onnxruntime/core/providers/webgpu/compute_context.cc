// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {
ComputeContext::ComputeContext(OpKernelContext& kernel_context, const WebGpuExecutionProvider& ep)
    : webgpu_context_{WebGpuContextFactory::GetContext(kernel_context.GetDeviceId())},
      kernel_context_{kernel_context},
      ep_{ep} {
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

const webgpu::BufferManager& ComputeContext::BufferManager() const {
  return ep_.BufferManager();
}

}  // namespace webgpu
}  // namespace onnxruntime
