// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/ngraph/ngraph_provider_factory.h"
#include <atomic>
#include "ngraph_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct NGraphProviderFactory : IExecutionProviderFactory {
  NGraphProviderFactory(std::string&& ng_backend_type) : ng_backend_type_(std::move(ng_backend_type)) {}
  ~NGraphProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    NGRAPHExecutionProviderInfo info{ng_backend_type_};
    return std::make_unique<NGRAPHExecutionProvider>(info);
  }

  private:
    const std::string ng_backend_type_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_NGraph(const char* ng_backend_type) {
  return std::make_shared<onnxruntime::NGraphProviderFactory>(std::string{ng_backend_type});
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_NGraph, _In_ OrtSessionOptions* options, _In_ const char* ng_backend_type) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_NGraph(ng_backend_type));
  return nullptr;
}
