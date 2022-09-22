// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <memory>

#include "core/providers/tvm/tvm_provider_factory.h"
#include "core/session/abi_session_options_impl.h"

#include "tvm_execution_provider.h"
#include "tvm_provider_factory_creator.h"
#include "tvm_so_execution_provider.h"  // NOLINT(build/include_subdir)

namespace onnxruntime {

struct TvmProviderFactory : IExecutionProviderFactory {
  TvmProviderFactory(const tvm::TvmEPOptions& options) : options_{options} {}
  ~TvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    std::unique_ptr<IExecutionProvider> provider = nullptr;
    if (options_.so_folder != "") {
      ORT_ENFORCE(options_.executor == "vm",
                  "Only virtual machine module is compiled from shared lib and dependences!");
      provider = std::move(std::make_unique<tvm::TvmSoExecutionProvider>(options_));
    } else {
      provider = std::move(std::make_unique<tvm::TvmExecutionProvider>(options_));
    }

    return provider;
  }

 private:
  tvm::TvmEPOptions options_;
};

std::shared_ptr<IExecutionProviderFactory> TVMProviderFactoryCreator::Create(const char* opt_str) {
  tvm::TvmEPOptions options = tvm::TvmEPOptionsHelper::FromOptionsString(opt_str);
  return std::make_shared<TvmProviderFactory>(options);
}

std::shared_ptr<IExecutionProviderFactory> TVMProviderFactoryCreator::Create(const tvm::TvmEPOptions& options) {
  return std::make_shared<TvmProviderFactory>(options);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tvm,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* opt_str) {
  onnxruntime::tvm::TvmEPOptions tvm_options = onnxruntime::tvm::TvmEPOptionsHelper::FromOptionsString(opt_str);
  options->provider_factories.push_back(onnxruntime::TVMProviderFactoryCreator::Create(tvm_options));
  return nullptr;
}
