// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/compute_context.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {

ComputeContextBase::ComputeContextBase(WebGpuContext& webgpu_context,
                                       const WebGpuExecutionProvider& ep,
                                       const OpKernel& op_kernel)
    : webgpu_context_{webgpu_context},
      ep_{ep},
      op_kernel_{op_kernel} {
}

const webgpu::BufferManager& ComputeContextBase::BufferManagerAccessor::Get(const ComputeContextBase& context) {
  return context.ep_.BufferManager();
}

ComputeContext::ComputeContext(WebGpuContext& webgpu_context,
                               const WebGpuExecutionProvider& ep,
                               const OpKernel& op_kernel,
                               OpKernelContext& kernel_context)
    : ComputeContextBase(webgpu_context, ep, op_kernel),
      kernel_context_{kernel_context} {
}

}  // namespace webgpu
}  // namespace onnxruntime
