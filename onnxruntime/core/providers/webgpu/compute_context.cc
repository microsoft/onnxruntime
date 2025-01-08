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
  if (webgpu_context_.ValidationMode() >= ValidationMode::Basic) {
    webgpu_context_.Device().PushErrorScope(wgpu::ErrorFilter::Validation);
  }
}

Status ComputeContext::PopErrorScope() {
  Status status{};

  if (webgpu_context_.ValidationMode() >= ValidationMode::Basic) {
    ORT_RETURN_IF_ERROR(webgpu_context_.Wait(
        webgpu_context_.Device().PopErrorScope(
            wgpu::CallbackMode::WaitAnyOnly, [](wgpu::PopErrorScopeStatus pop_status, wgpu::ErrorType error_type, char const* message, Status* status) {
              ORT_ENFORCE(pop_status == wgpu::PopErrorScopeStatus::Success, "Instance dropped.");
              if (error_type == wgpu::ErrorType::NoError) {
                return;
              }
              *status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "WebGPU validation failed. ", message);
            },
            &status)));
  }
  return status;
}

}  // namespace webgpu
}  // namespace onnxruntime
