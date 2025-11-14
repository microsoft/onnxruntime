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
      ep_(*static_cast<const WebGpuExecutionProvider*>(info.GetExecutionProvider())) {
}

Status WebGpuKernel::Compute(OpKernelContext* p_op_kernel_context) const {
  WebGpuContext& webgpu_context = WebGpuContextFactory::GetContext(ep_.GetDeviceId());
  ComputeContext context{*p_op_kernel_context, ep_, webgpu_context};

  if (webgpu_context.ValidationMode() >= ValidationMode::Full) {
    webgpu_context.PushErrorScope();
  }

  Status s = ComputeInternal(context);

  if (webgpu_context.ValidationMode() >= ValidationMode::Full) {
    ORT_RETURN_IF_ERROR(webgpu_context.PopErrorScope());
  }

  return s;
}

}  // namespace webgpu
}  // namespace onnxruntime
