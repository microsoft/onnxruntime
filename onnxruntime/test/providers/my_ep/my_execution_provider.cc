// Copyright (c) Microsoft Corporation. All rights reserved.
// Confidential and Proprietary.

#include "my_execution_provider.h"
#include "my_ep_allocator.h"

namespace onnxruntime {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMyProvider, kOnnxDomain, 13, Add);

static std::shared_ptr<KernelRegistry> RegisterCudaKernels() {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMyProvider, kOnnxDomain, 13, Add)>};
  static std::shared_ptr<KernelRegistry> kernel_registry =
      KernelRegistry::Create();
  for (auto& function : function_table) {
    KernelCreateInfo info = function();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      if (!kernel_registry->Register(std::move(info)).IsOK()) {
        ORT_THROW("Failed to register kernel");
      }
    }
  }
  return kernel_registry;
}

std::shared_ptr<KernelRegistry> MyExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = RegisterCudaKernels();
  return kernel_registry;
}

MyExecutionProvider::MyExecutionProvider(const MyProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMyProvider}, device_id_(info.device_id) {
  AllocatorCreationInfo device_info{
      [](OrtDevice::DeviceId device_id) { return std::make_unique<MyEPAllocator>(device_id); },
      device_id_,
      true,
      {0, 1, -1, -1, -1}};
  InsertAllocator(CreateAllocator(device_info));
}

}  // namespace onnxruntime
