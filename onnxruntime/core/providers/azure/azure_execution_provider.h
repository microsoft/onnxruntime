// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

struct AzureExecutionProviderInfo {
  std::string end_point;
  std::string access_token;
};

class AzureExecutionProvider : public IExecutionProvider {
 public:
  AzureExecutionProvider(const AzureExecutionProviderInfo& azure_ep_info) : IExecutionProvider{onnxruntime::kAzureExecutionProvider}, info(azure_ep_info) {
    AllocatorCreationInfo device_info{[](int) { return std::make_unique<CPUAllocator>(); },
                                      DEFAULT_CPU_ALLOCATOR_DEVICE_ID, false};
    InsertAllocator(CreateAllocator(device_info));
  }
  void RegisterAllocator(AllocatorManager& allocator_manager) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

  const AzureExecutionProviderInfo info;
};

}  // namespace onnxruntime