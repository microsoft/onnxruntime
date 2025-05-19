// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_provider_factory_creator.h"

#include <algorithm>
#include <cctype>
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
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

 private:
  ProviderOptions info_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  return std::make_unique<VitisAIExecutionProvider>(info_);
}

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                                                           const OrtLogger& session_logger) {
  const ConfigOptions& config_options = session_options.GetConfigOptions();
  const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  // Extract those EP options into a new "provider_options" map.
  std::string lowercase_ep_name = kVitisAIExecutionProvider;
  std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  std::string key_prefix = "ep.";
  key_prefix += lowercase_ep_name;
  key_prefix += ".";

  std::unordered_map<std::string, std::string> provider_options = info_;
  for (const auto& [key, value] : config_options_map) {
    if (key.rfind(key_prefix, 0) == 0) {
      provider_options[key.substr(key_prefix.size())] = value;
    } else {
      provider_options["ort_session_config." + key] = value;
    }
  }

  // Store pointer to session options as done in SessionOptionsAppendExecutionProvider_VitisAI
  provider_options["session_options"] = std::to_string((uintptr_t)(void*)&session_options);

  auto ep_instance = std::make_unique<VitisAIExecutionProvider>(provider_options);
  ep_instance->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
  return ep_instance;
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
  void GetCustomOpDomainList(IExecutionProviderFactory*, std::vector<OrtCustomOpDomain*>&) override {};
  // Called right after loading the shared library, if this throws any errors Shutdown() will be called and the library unloaded
  void Initialize() override { initialize_vitisai_ep(); }
  // Called right before unloading the shared library
  void Shutdown() override { deinitialize_vitisai_ep(); }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
