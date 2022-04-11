// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <memory>

#include "core/providers/tvm/tvm_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "tvm_execution_provider.h"


namespace onnxruntime {

struct TvmProviderFactory : IExecutionProviderFactory {
  TvmProviderFactory(const tvm::TvmEPOptions& options) : options_{options} {}
  ~TvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<tvm::TvmExecutionProvider>(options_);
  }

private:
  tvm::TvmEPOptions options_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char* opt_str) {
  tvm::TvmEPOptions options = tvm::TvmEPOptionsHelper::FromOptionsString(opt_str);
  return std::make_shared<TvmProviderFactory>(options);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const tvm::TvmEPOptions& options) {
  return std::make_shared<TvmProviderFactory>(options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tvm,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* opt_str) {
  onnxruntime::tvm::TvmEPOptions tvm_options = onnxruntime::tvm::TvmEPOptionsHelper::FromOptionsString(opt_str);
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tvm(tvm_options));
  return nullptr;
}
