// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <unordered_map>
#include "core/providers/azure/azure_provider_factory.h"
#include "core/providers/azure/azure_provider_factory_creator.h"
#include "core/providers/azure/azure_execution_provider.h"

namespace onnxruntime {

AzureProviderFactory::AzureProviderFactory(const std::unordered_map<std::string, std::string>& config) : config_(config) {}

std::unique_ptr<IExecutionProvider> AzureProviderFactory::CreateProvider() {
  return std::make_unique<AzureExecutionProvider>(config_);
}

std::shared_ptr<IExecutionProviderFactory> AzureProviderFactoryCreator::Create(const std::unordered_map<std::string, std::string>& config) {
  return std::make_shared<AzureProviderFactory>(config);
}

}  // namespace onnxruntime