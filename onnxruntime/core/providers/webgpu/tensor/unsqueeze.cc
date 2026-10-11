// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/unsqueeze.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

KernelCreateInfo CreateUnsqueezeVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, true);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Unsqueeze>(info);
    return Status::OK();
  };

  if (start_version >= 13) {
    return {
        KernelDefBuilder()
            .SetName("Unsqueeze")
            .SetDomain(kOnnxDomain)
            .SinceVersion(start_version, end_version)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T", type_constraints)
            .Alias(0, 0)
            .InputMemoryType(OrtMemTypeCPU, 1)
            .Build(),
        kernel_create_fn};
  } else {
    return {
        KernelDefBuilder()
            .SetName("Unsqueeze")
            .SetDomain(kOnnxDomain)
            .SinceVersion(start_version, end_version)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T", type_constraints)
            .Alias(0, 0)
            .Build(),
        kernel_create_fn};
  }
}

KernelCreateInfo CreateUnsqueezeKernelInfo(int since_version, bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, true);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Unsqueeze>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Unsqueeze")
          .SetDomain(kOnnxDomain)
          .SinceVersion(since_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

}  // namespace webgpu
}  // namespace onnxruntime
