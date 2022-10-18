// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cloud/cloud_execution_provider.h"

namespace onnxruntime {

CloudExecutionProvider::CloudExecutionProvider(const CloudExecutionProviderInfo& info) : IExecutionProvider{onnxruntime::kCloudExecutionProvider}, info_(info) {
}

CloudExecutionProvider::~CloudExecutionProvider() {
}

std::shared_ptr<KernelRegistry> CloudExecutionProvider::GetKernelRegistry() const {
  return nullptr;
}

}  // namespace onnxruntime