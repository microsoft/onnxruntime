// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shl/shl_provider_factory.h"
#include "shl_execution_provider.h"
#include "shl_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ShlProviderFactory : public IExecutionProviderFactory {
  ShlProviderFactory(const std::unordered_map<std::string, std::string>& config) : config_(config) {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unordered_map<std::string, std::string> config_;
};

std::unique_ptr<IExecutionProvider> ShlProviderFactory ::CreateProvider() {
  return std::make_unique<ShlExecutionProvider>(config_);
};

std::shared_ptr<IExecutionProviderFactory> ShlProviderFactoryCreator::Create(const std::unordered_map<std::string, std::string>& config) {
  return std::make_shared<ShlProviderFactory>(config);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Shl, _In_ OrtSessionOptions* options, const std::unordered_map<std::string, std::string>& config) {
  options->provider_factories.push_back(onnxruntime::ShlProviderFactoryCreator::Create(config));
  return nullptr;
}
