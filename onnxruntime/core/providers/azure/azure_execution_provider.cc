// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/azure/azure_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAzureExecutionProvider, kMSDomain, 1, RemoteCall);

Status RegisterAzureEPKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAzureExecutionProvider, kMSDomain, 1, RemoteCall)>,
  };
  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
  }
  return Status::OK();
}

void AzureExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice cpu_device{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
  auto cpu_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, cpu_device);
  if (!cpu_alloc) {
    allocator_manager.InsertAllocator(GetAllocator(cpu_device.Id(), OrtMemTypeDefault));
  }
}

std::unique_ptr<IDataTransfer> AzureExecutionProvider::GetDataTransfer() const {
  return std::make_unique<CPUDataTransfer>();
}

std::shared_ptr<KernelRegistry> AzureExecutionProvider::GetKernelRegistry() const {
  //todo - find out if beneficial make k to be static?
  std::shared_ptr<KernelRegistry> k = std::make_shared<KernelRegistry>();
  ORT_ENFORCE(RegisterAzureEPKernels(*k).IsOK(), "Failed to initialize AzureEP kernel register");
  return k;
}

}