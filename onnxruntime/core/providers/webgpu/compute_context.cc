// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"

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

const webgpu::BufferManager& ComputeContext::BufferManager() const {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0);
  AllocatorPtr allocator = kernel_context_.GetAllocator(gpu_device);
  const GpuBufferAllocator* gpu_allocator = static_cast<const GpuBufferAllocator*>(allocator.get());
  return gpu_allocator->GetBufferManager();
}

}  // namespace webgpu
}  // namespace onnxruntime
