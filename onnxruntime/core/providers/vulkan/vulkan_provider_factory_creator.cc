// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_execution_provider.h"
#include "core/providers/vulkan/vulkan_provider_factory_creator.h"

namespace onnxruntime {

struct VulkanProviderFactory : IExecutionProviderFactory {
  VulkanProviderFactory(const ProviderOptions& provider_options)
      : info_{provider_options} {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<VulkanExecutionProvider>(info_);
  }

 private:
  VulkanExecutionProviderInfo info_;
};
  std::shared_ptr<IExecutionProviderFactory> VulkanProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  return std::make_shared<VulkanProviderFactory>(provider_options);
}

}  // namespace onnxruntime
