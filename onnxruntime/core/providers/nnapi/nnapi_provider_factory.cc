// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/nnapi/nnapi_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "nnapi_builtin/nnapi_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct NnapiProviderFactory : IExecutionProviderFactory {
  NnapiProviderFactory(uint32_t nnapi_flags)
      : nnapi_flags_(nnapi_flags) {}
  ~NnapiProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  uint32_t nnapi_flags_;
};

std::unique_ptr<IExecutionProvider> NnapiProviderFactory::CreateProvider() {
  return std::make_unique<NnapiExecutionProvider>(nnapi_flags_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(uint32_t nnapi_flags) {
  return std::make_shared<onnxruntime::NnapiProviderFactory>(nnapi_flags);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Nnapi(nnapi_flags));
  return nullptr;
}
