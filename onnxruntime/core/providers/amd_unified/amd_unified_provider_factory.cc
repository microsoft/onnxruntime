// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./amd_unified_provider_factory.h"
#include "./amd_unified_provider_factory_creator.h"
#include "./amd_unified_execution_provider.h"

// 1st-party libs/headers.
#include "core/providers/shared_library/provider_api.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

// 3rd-party libs/headers.
#include "nlohmann/json.hpp"

// Standard libs/headers.
#include <fstream>
#include <unordered_map>
#include <string>


using namespace onnxruntime;
using json = nlohmann::json;

namespace onnxruntime {

#if 0
void InitializeRegistry();
void DeleteRegistry();
#endif

struct AMDUnifiedProviderFactory : IExecutionProviderFactory {
  AMDUnifiedProviderFactory(const AMDUnifiedExecutionProviderInfo& ep_info)
    : ep_info_(ep_info) {}
  ~AMDUnifiedProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  AMDUnifiedExecutionProviderInfo ep_info_;
};

std::unique_ptr<IExecutionProvider> AMDUnifiedProviderFactory::CreateProvider() {
  return std::make_unique<AMDUnifiedExecutionProvider>(ep_info_);
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_AMDUnified(
    const AMDUnifiedExecutionProviderInfo& ep_info) {
  return std::make_shared<AMDUnifiedExecutionProvider>(ep_info);
}

std::shared_ptr<IExecutionProviderFactory>
AMDUnifiedProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  auto ep_info = AMDUnifiedExecutionProviderInfo{provider_options};
  return std::shared_ptr<AMDUnifiedProviderFactory>(ep_info);
}

#if 0
struct AMDUnified_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      int device_id) override {}

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      const void* provider_options) override {}

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }
};
#endif

}  // namespace onnxruntime
