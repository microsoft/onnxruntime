// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/dnnl/dnnl_provider_factory.h"
#include <atomic>
#include "dnnl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct DnnlProviderFactory : IExecutionProviderFactory {
  DnnlProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~DnnlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  DNNLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<DNNLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(bool use_arena) {
  return std::make_shared<onnxruntime::DnnlProviderFactory>(use_arena);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Dnnl(bool(use_arena)));
  return nullptr;
}
