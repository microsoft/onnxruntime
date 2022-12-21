// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cloud/cloud_execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {
CloudExecutionProvider::CloudExecutionProvider(const std::unordered_map<std::string, std::string>& config) : IExecutionProvider{onnxruntime::kCloudExecutionProvider},
                                                                                                             config_(config) {
}
}  // namespace onnxruntime