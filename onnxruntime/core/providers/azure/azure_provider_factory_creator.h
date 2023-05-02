// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include "core/providers/providers.h"

namespace onnxruntime {

struct AzureProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const std::unordered_map<std::string, std::string>& config);
};

}  // namespace onnxruntime
