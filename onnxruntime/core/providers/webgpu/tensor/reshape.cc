// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/reshape.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateReshapeVersionedKernelInfo(bool enable_int64) {
  // Reshape is a pure copy/view op. Enabling int64 is safe because element values are never
  // interpreted or used in shader arithmetic.
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, true);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Reshape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Reshape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(StartVersion, EndVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

template <int SinceVersion>
KernelCreateInfo CreateReshapeKernelInfo(bool enable_int64) {
  // Reshape is a pure copy/view op. Enabling int64 is safe because element values are never
  // interpreted or used in shader arithmetic.
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, true);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Reshape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Reshape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(SinceVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

// Explicit template instantiations
template KernelCreateInfo CreateReshapeVersionedKernelInfo<5, 12>(bool);
template KernelCreateInfo CreateReshapeVersionedKernelInfo<13, 13>(bool);
template KernelCreateInfo CreateReshapeVersionedKernelInfo<14, 18>(bool);
template KernelCreateInfo CreateReshapeVersionedKernelInfo<19, 20>(bool);
template KernelCreateInfo CreateReshapeVersionedKernelInfo<21, 22>(bool);
template KernelCreateInfo CreateReshapeVersionedKernelInfo<23, 24>(bool);
template KernelCreateInfo CreateReshapeKernelInfo<25>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
