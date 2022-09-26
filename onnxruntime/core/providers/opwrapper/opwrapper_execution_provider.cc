// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opwrapper/opwrapper_execution_provider.h"
#include <vector>
#include <memory>
#include <utility>
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

OpWrapperExecutionProvider::OpWrapperExecutionProvider(const ProviderOptions& provider_options)
    : IExecutionProvider{onnxruntime::kOpWrapperExecutionProvider}, provider_options_{provider_options} {
  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo("OpWrapper", OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

OpWrapperExecutionProvider::~OpWrapperExecutionProvider() {}

}  // namespace onnxruntime