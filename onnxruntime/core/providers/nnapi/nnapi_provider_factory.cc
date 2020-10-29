// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/nnapi/nnapi_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "nnapi_builtin/nnapi_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct NnapiProviderFactory : IExecutionProviderFactory {
  NnapiProviderFactory() {}
  ~NnapiProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> NnapiProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<NnapiExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi() {
  return std::make_shared<onnxruntime::NnapiProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Nnapi());
  return nullptr;
}
