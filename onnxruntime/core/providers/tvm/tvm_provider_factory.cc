// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <memory>

#include "core/providers/tvm/tvm_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "tvm_execution_provider.h"


namespace onnxruntime {

struct TvmProviderFactory : IExecutionProviderFactory {
  TvmProviderFactory(const TvmExecutionProviderInfo& info) : info_{info} {}
  ~TvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<TvmExecutionProvider>(info_);
  }

 private:
    TvmExecutionProviderInfo info_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char* settings) {
    TvmExecutionProviderInfo info = TvmExecutionProviderInfo::FromOptionsString(settings);
    return std::make_shared<TvmProviderFactory>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const TvmExecutionProviderInfo& info)
{
    return std::make_shared<TvmProviderFactory>(info);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tvm,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* settings) {
  onnxruntime::TvmExecutionProviderInfo info = onnxruntime::TvmExecutionProviderInfo::FromOptionsString(settings);
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tvm(info));
  return nullptr;
}
