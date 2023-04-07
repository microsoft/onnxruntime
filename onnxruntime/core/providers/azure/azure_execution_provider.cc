// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/azure/azure_execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {
AzureExecutionProvider::AzureExecutionProvider(const std::unordered_map<std::string, std::string>& config) : IExecutionProvider{onnxruntime::kAzureExecutionProvider}, config_(config) {
}
}  // namespace onnxruntime