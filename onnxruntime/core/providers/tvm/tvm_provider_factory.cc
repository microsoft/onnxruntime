// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <memory>

#include "core/providers/tvm/tvm_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "tvm_execution_provider.h"


namespace onnxruntime {

struct TvmProviderFactory : IExecutionProviderFactory {
  TvmProviderFactory(const TvmEPOptions& options) : options_{options} {}
  ~TvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<TvmExecutionProvider>(options_);
  }

 private:
    TvmEPOptions options_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char* opt_str) {
    TvmEPOptions options = TvmEPOptions::FromOptionsString(opt_str);
    return std::make_shared<TvmProviderFactory>(options);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const TvmEPOptions& options)
{
    return std::make_shared<TvmProviderFactory>(options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tvm,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* opt_str) {
  onnxruntime::TvmEPOptions tvm_options = onnxruntime::TvmEPOptions::FromOptionsString(opt_str);
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tvm(tvm_options));
  return nullptr;
}
