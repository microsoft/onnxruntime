// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include "core/providers/cloud/cloud_provider_factory_creator.h"
#include "core/providers/cloud/cloud_execution_provider.h"

namespace onnxruntime {

struct CloudProviderFactory : public IExecutionProviderFactory {
  CloudProviderFactory(const std::unordered_map<std::string, std::string>& config) : config_(config){}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<CloudExecutionProvider>(config_);
  };

  std::unordered_map<std::string, std::string> config_;
};

std::shared_ptr<IExecutionProviderFactory> CloudProviderFactoryCreator::Create(const std::unordered_map<std::string, std::string>& config) {
  return std::make_shared<CloudProviderFactory>(config);
}

}  // namespace onnxruntime