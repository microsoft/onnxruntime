// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/webnn_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "webnn_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct WebNNProviderFactory : IExecutionProviderFactory {
  WebNNProviderFactory(uint32_t webnn_device_flags, uint32_t webnn_power_flags)
      : webnn_device_flags_(webnn_device_flags), webnn_power_flags_(webnn_power_flags) {}
  ~WebNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  uint32_t webnn_device_flags_;
  uint32_t webnn_power_flags_;
};

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider() {
  return std::make_unique<WebNNExecutionProvider>(webnn_device_flags_, webnn_power_flags_);
}

std::shared_ptr<IExecutionProviderFactory> WebNNProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  uint32_t webnn_device_flags = 2, webnn_power_flags = 0;
  if (auto it = provider_options.find("deviceType"); it != provider_options.end()) {
    webnn_device_flags = std::stoi(it->second);
  }
  if (auto it = provider_options.find("powerPreference"); it != provider_options.end()) {
    webnn_power_flags = std::stoi(it->second);
  }
  return std::make_shared<onnxruntime::WebNNProviderFactory>(webnn_device_flags, webnn_power_flags);
}

}  // namespace onnxruntime
