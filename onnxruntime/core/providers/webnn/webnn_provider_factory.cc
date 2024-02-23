// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/webnn_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "webnn_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct WebNNProviderFactory : IExecutionProviderFactory {
  WebNNProviderFactory(const std::string& webnn_device_flags, const std::string& webnn_threads_number,
                       const std::string& webnn_power_flags)
      : webnn_device_flags_(webnn_device_flags), webnn_threads_number_(webnn_threads_number), webnn_power_flags_(webnn_power_flags) {}
  ~WebNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  std::string webnn_device_flags_;
  std::string webnn_threads_number_;
  std::string webnn_power_flags_;
};

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider() {
  return std::make_unique<WebNNExecutionProvider>(webnn_device_flags_, webnn_threads_number_, webnn_power_flags_);
}

std::shared_ptr<IExecutionProviderFactory> WebNNProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  return std::make_shared<onnxruntime::WebNNProviderFactory>(provider_options.at("deviceType"),
                                                             provider_options.at("numThreads"),
                                                             provider_options.at("powerPreference"));
}

}  // namespace onnxruntime
