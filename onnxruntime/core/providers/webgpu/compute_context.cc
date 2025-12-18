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

Status ComputeContextBase::CreateUnmappedGPUTensor(AllocatorPtr alloc, MLDataType data_type, const TensorShape& shape, std::unique_ptr<Tensor>& tensor) const {
  ORT_RETURN_IF_NOT(alloc != nullptr, "Allocator must not be null when creating GPU tensor.");

  tensor = std::make_unique<Tensor>(data_type, shape, alloc);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Failed to allocate GPU tensor.");

  void* data = tensor->MutableDataRaw();
  ORT_RETURN_IF_NOT(data != nullptr, "Failed to get GPU tensor buffer.");

  auto buffer = reinterpret_cast<WGPUBuffer>(data);
  if (wgpuBufferGetMapState(buffer) != WGPUBufferMapState_Unmapped) {
    wgpuBufferUnmap(buffer);
  }
  return Status::OK();
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
