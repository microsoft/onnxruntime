// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {
ComputeContext::ComputeContext(OpKernelContext& kernel_context, const WebGpuExecutionProvider& ep, WebGpuContext& webgpu_context)
    : webgpu_context_{webgpu_context},
      kernel_context_{kernel_context},
      ep_{ep} {
}

const webgpu::BufferManager& ComputeContext::BufferManagerAccessor::Get(const ComputeContext& context) {
  return context.ep_.BufferManager();
}

}  // namespace webgpu
}  // namespace onnxruntime
