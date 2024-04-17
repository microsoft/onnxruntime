// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_provider_factory_creator.h"

#include <unordered_map>
#include <string>

#include "vaip/global_api.h"
#include "./vitisai_execution_provider.h"
#include "core/framework/execution_provider.h"

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

struct VitisAI_Provider : Provider {
  // Takes a pointer to a provider specific structure to create the factory. For example, with OpenVINO it is a pointer to an OrtOpenVINOProviderOptions structure
  std::shared_ptr<IExecutionProviderFactory>
  CreateExecutionProviderFactory(const void* options) override {
    return std::make_shared<VitisAIProviderFactory>(GetProviderOptions(options));
  }
  // Convert provider options struct to ProviderOptions which is a map
  ProviderOptions GetProviderOptions(const void* options) override {
    auto vitisai_options = reinterpret_cast<const ProviderOptions*>(options);
    return *vitisai_options;
  }
  // Update provider options from key-value string configuration
  void UpdateProviderOptions(void* options, const ProviderOptions& provider_options) override {
    auto vitisai_options = reinterpret_cast<ProviderOptions*>(options);
    for (const auto& entry : provider_options) {
      vitisai_options->insert_or_assign(entry.first, entry.second);
    }
  };
  // Get provider specific custom op domain list. Provider has the resposibility to release OrtCustomOpDomain instances it creates.
  void GetCustomOpDomainList(IExecutionProviderFactory*, std::vector<OrtCustomOpDomain*>&) override{};
  // Called right after loading the shared library, if this throws any errors Shutdown() will be called and the library unloaded
  void Initialize() override { initialize_vitisai_ep(); }
  // Called right before unloading the shared library
  void Shutdown() override {}
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
