// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/providers/opwrapper/opwrapper_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"

namespace onnxruntime {

OpWrapperExecutionProvider::OpWrapperExecutionProvider(const ProviderOptionsMap& provider_options_map)
    : IExecutionProvider{onnxruntime::kOpWrapperExecutionProvider}, provider_options_map_{provider_options_map} {
  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo("OpWrapper", OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

OpWrapperExecutionProvider::~OpWrapperExecutionProvider() {}

ProviderOptions OpWrapperExecutionProvider::GetOpProviderOptions(const std::string& op_name) const {
  auto it = provider_options_map_.find(op_name);
  return (it == provider_options_map_.end()) ? ProviderOptions{} : it->second;
}

}  // namespace onnxruntime
