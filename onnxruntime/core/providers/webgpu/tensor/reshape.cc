// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/reshape.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

KernelCreateInfo CreateReshapeVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  // Reshape is a pure copy/view op. Enabling int64 and uint8 are safe because element values
  // are never interpreted or used in shader arithmetic.
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, true);
  type_constraints.push_back(DataTypeImpl::GetTensorType<uint8_t>());

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Reshape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Reshape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(start_version, end_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

KernelCreateInfo CreateReshapeKernelInfo(int since_version, bool enable_int64) {
  // Reshape is a pure copy/view op. Enabling int64 and uint8 are safe because element values
  // are never interpreted or used in shader arithmetic.
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, true);
  type_constraints.push_back(DataTypeImpl::GetTensorType<uint8_t>());

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Reshape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Reshape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(since_version)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", std::move(type_constraints))
          .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
          .Alias(0, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .Build(),
      kernel_create_fn};
}

}  // namespace webgpu
}  // namespace onnxruntime
