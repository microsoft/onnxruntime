// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/xnnpack_provider_factory.h"

#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const SessionOptions* options) override;
  XnnpackExecutionProviderInfo info{true, 1};
};

std::unique_ptr<IExecutionProvider> XnnpackProviderFactory::CreateProvider(const SessionOptions* options) {
  info.xnn_thread_pool_size = options->intra_op_param.thread_pool_size;
  return CreateProvider();
}

std::unique_ptr<IExecutionProvider> XnnpackProviderFactory::CreateProvider() {
  return std::make_unique<XnnpackExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Xnnpack(OrtXnnpackProviderOptions* /*ep_options*/) {
  auto factory = std::make_shared<XnnpackProviderFactory>();
  // factory->info.options = ep_options == 0 ? OrtXnnpackProviderOptions() : ep_options;
  return factory;
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Xnnpack, _In_ OrtSessionOptions* options,
	_In_ OrtXnnpackProviderOptions* ep_options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Xnnpack(ep_options));
  return nullptr;
}
