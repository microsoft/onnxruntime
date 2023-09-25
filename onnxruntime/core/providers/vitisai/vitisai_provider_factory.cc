// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_provider_factory_creator.h"

#include "vaip/global_api.h"
#include "./vitisai_execution_provider.h"
#include "core/framework/execution_provider.h"

#include "core/session/abi_session_options_impl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <unordered_map>
#include <string>

using namespace onnxruntime;
using json = nlohmann::json;
namespace onnxruntime {

static std::string ConfigToJsonStr(const std::unordered_map<std::string, std::string>& config) {
  const auto& filename = config.at("config_file");
  std::ifstream f(filename);
  json data = json::parse(f);
  for (const auto& entry : config) {
    data[entry.first] = entry.second;
  }
  return data.dump();
}

VitisAIExecutionProviderInfo::VitisAIExecutionProviderInfo(const ProviderOptions& provider_options) : provider_options_(provider_options), json_config_{ConfigToJsonStr(provider_options)} {}

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

std::shared_ptr<IExecutionProviderFactory> VitisAIProviderFactoryCreator::Create(const ProviderOptions& provider_options) {
  initialize_vitisai_ep();
  auto info = VitisAIExecutionProviderInfo{provider_options};
  return std::make_shared<VitisAIProviderFactory>(info);
}

}  // namespace onnxruntime
