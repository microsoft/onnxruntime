// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_provider_factory_creator.h"

#include "vaip/global_api.h"
#include "./vitisai_execution_provider.h"
#include "core/framework/execution_provider.h"

#include "core/session/abi_session_options_impl.h"
#include "core/providers/shared_library/provider_host_api.h"
#include <unordered_map>
#include <string>

using namespace onnxruntime;
namespace onnxruntime {

struct VitisAIProviderFactory : IExecutionProviderFactory {
  VitisAIProviderFactory(const ProviderOptions& info) : info_(info) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  ProviderOptions info_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  return std::make_unique<VitisAIExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> VitisAIProviderFactoryCreator::Create(const ProviderOptions& provider_options) {
  initialize_vitisai_ep();
  return std::make_shared<VitisAIProviderFactory>(provider_options);
}

}  // namespace onnxruntime
