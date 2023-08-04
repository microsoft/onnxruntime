// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/providers.h"

namespace onnxruntime {

struct AzureProviderFactory : public IExecutionProviderFactory {
  AzureProviderFactory(const std::unordered_map<std::string, std::string>& config);
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unordered_map<std::string, std::string> config_;
};

}  // namespace onnxruntime