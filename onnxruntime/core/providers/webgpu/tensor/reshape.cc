// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/reshape.h"
#include "core/framework/int4.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Reshape is a pure copy/view op. Enabling int64, uint8 and the 4-bit types is safe because element
// values are never interpreted or used in shader arithmetic. The schema allows int4/uint4 from opset 21.
std::vector<MLDataType> GetReshapeTypeConstraints(int since_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetOpTypeConstraints(enable_int64, true);
  type_constraints.push_back(DataTypeImpl::GetTensorType<uint8_t>());
  if (since_version >= 21) {
    type_constraints.push_back(DataTypeImpl::GetTensorType<Int4x2>());
    type_constraints.push_back(DataTypeImpl::GetTensorType<UInt4x2>());
  }
  return type_constraints;
}

}  // namespace

KernelCreateInfo CreateReshapeVersionedKernelInfo(int start_version, int end_version, bool enable_int64) {
  std::vector<MLDataType> type_constraints = GetReshapeTypeConstraints(start_version, enable_int64);

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
  std::vector<MLDataType> type_constraints = GetReshapeTypeConstraints(since_version, enable_int64);

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
