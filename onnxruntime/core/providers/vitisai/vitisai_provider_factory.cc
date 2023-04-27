// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vitisai/vitisai_provider_factory.h"
#include "vitisai_provider_factory_creator.h"

#include "vaip/global_api.hpp"
#include "./vitisai_execution_provider.h"
#include "core/framework/execution_provider.h"

#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct VitisAIProviderFactory : IExecutionProviderFactory {
  VitisAIProviderFactory(const VitisAIExecutionProviderInfo& info) : info_(info) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  VitisAIExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  return std::make_unique<VitisAIExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_VITISAI(const VitisAIExecutionProviderInfo& info) {
  initialize_vitisai_ep();
  return std::make_shared<VitisAIProviderFactory>(info);
}

std::shared_ptr<IExecutionProviderFactory> VitisAIProviderFactoryCreator::Create(const VitisAIExecutionProviderInfo& info) {
  initialize_vitisai_ep();
  return std::make_shared<VitisAIProviderFactory>(info);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_VITISAI,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* opt_str = nullptr) {
  auto info = VitisAIExecutionProviderInfo(opt_str);
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_VITISAI(info));
  return nullptr;
}
