// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/trt/trt_provider_factory.h"
#include <atomic>
#include "trt_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TRTProviderFactory : IExecutionProviderFactory {
  TRTProviderFactory() {}
  ~TRTProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> TRTProviderFactory::CreateProvider() {
  return std::make_unique<TRTExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_TRT() {
  return std::make_shared<onnxruntime::TRTProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_TRT, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_TRT());
  return nullptr;
}

