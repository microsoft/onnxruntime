// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/unsqueeze.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateUnsqueezeVersionedKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Unsqueeze>(info);
    return Status::OK();
  };

  if constexpr (StartVersion >= 13) {
    return {
        KernelDefBuilder()
            .SetName("Unsqueeze")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion, EndVersion)
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
            .SinceVersion(StartVersion, EndVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T", type_constraints)
            .Alias(0, 0)
            .Build(),
        kernel_create_fn};
  }
}

template <int SinceVersion>
KernelCreateInfo CreateUnsqueezeKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Unsqueeze>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Unsqueeze")
          .SetDomain(kOnnxDomain)
          .SinceVersion(SinceVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

// Explicit template instantiations
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<1, 10>(bool);
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<11, 12>(bool);
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<13, 20>(bool);
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<21, 22>(bool);
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<23, 23>(bool);
template KernelCreateInfo CreateUnsqueezeVersionedKernelInfo<24, 24>(bool);
template KernelCreateInfo CreateUnsqueezeKernelInfo<25>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
