// Copyright (c) Microsoft Corporation. All rights reserved.
// Confidential and Proprietary.

#include "my_execution_provider.h"
#include "my_allocator.h"

namespace onnxruntime {

std::shared_ptr<KernelRegistry>
MyExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry =
      KernelRegistry::Create();
  return kernel_registry;
}

MyExecutionProvider::MyExecutionProvider(const MyProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMyProvider}, device_id_(info.device_id) {
  
  // Athena emulator currently has ~1.2 GB of HBM for user data.  Reserve 1 GB for now.
  const size_t max_mem = 1 * 1024 * 1024 * 1024;
  AllocatorCreationInfo device_info{
      [](OrtDevice::DeviceId device_id) { return std::make_unique<MyEPAllocator>(device_id); },
      device_id_,
      true,
      {0, 1, -1, -1}};
  InsertAllocator(CreateAllocator(device_info));
}

}  // namespace onnxruntime
