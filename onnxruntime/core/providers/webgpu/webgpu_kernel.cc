// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

WebGpuKernel::WebGpuKernel(const OpKernelInfo& info)
    : OpKernel(info),
      ep_(*static_cast<const WebGpuExecutionProvider*>(info.GetExecutionProvider())),
      webgpu_context_(WebGpuContextFactory::GetContext(ep_.GetDeviceId())) {
}

Status WebGpuKernel::Compute(OpKernelContext* p_op_kernel_context) const {
  ComputeContext context{webgpu_context_,
                         ep_,
                         *this,
                         *p_op_kernel_context};

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    webgpu_context_.PushErrorScope();
  }

  Status s = ComputeInternal(context);

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    ORT_RETURN_IF_ERROR(webgpu_context_.PopErrorScope());
  }

  return s;
}

Status WebGpuKernel::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                             /*out*/ bool& is_packed, /*out*/ PrePackedWeights* /* prepacked_weights */) {
  ComputeContextBase context{webgpu_context_, ep_, *this};

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    webgpu_context_.PushErrorScope();
  }

  // Currently, ORT does not allow using prepacked weights in non-CPU EPs.
  // So we do not pass prepacked_weights to PrePackInternal.
  // Kernel implementation that supports prepacking should manage its own storage.
  // Use the EP's prepack allocator which creates unmapped GPU buffers.

  Status s = PrePackInternal(context, tensor, input_idx, ep_.PrepackAllocator(), is_packed);

  if (is_packed) {
    // Flush pending commands to ensure GPU buffer creations are completed.
    // This allows the initializer buffer manager to release temporary buffers and reduce memory usage.
    webgpu_context_.Flush(webgpu_context_.InitializerBufferManager());
  }

  if (webgpu_context_.ValidationMode() >= ValidationMode::Full) {
    ORT_RETURN_IF_ERROR(webgpu_context_.PopErrorScope());
  }

  return s;
}

Status WebGpuKernel::PrePackInternal(ComputeContextBase& /*context*/,
                                     const Tensor& /*tensor*/,
                                     int /*input_idx*/,
                                     AllocatorPtr /*alloc*/,
                                     /*out*/ bool& is_packed) {
  is_packed = false;
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
