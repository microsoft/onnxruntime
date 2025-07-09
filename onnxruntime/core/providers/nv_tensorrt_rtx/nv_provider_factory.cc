// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "nv_provider_factory.h"
#include <atomic>
#include "nv_execution_provider.h"
#include "nv_provider_factory_creator.h"
#include "core/framework/provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_custom_ops.h"
#include <string.h>

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_Nv_Impl final : ProviderInfo_Nv {
  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) override {
    common::Status status = CreateTensorRTCustomOpDomainList(domain_list, extra_plugin_lib_paths);
    if (!status.IsOK()) {
      return CreateStatus(ORT_FAIL, "[NvTensorRTRTX EP] Can't create custom ops for TRT plugins.");
    }
    return nullptr;
  }

  OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) override {
    ReleaseTensorRTCustomOpDomainList(domain_list);
    return nullptr;
  }
} g_info;

struct NvProviderFactory : IExecutionProviderFactory {
  NvProviderFactory(const NvExecutionProviderInfo& info) : info_{info} {}
  ~NvProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger);

 private:
  NvExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider() {
  return std::make_unique<NvExecutionProvider>(info_);
}

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider(const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
  const ConfigOptions& config_options = session_options.GetConfigOptions();
  const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  // We extract those EP options to create a new "provider options" key/value map.
  std::string lowercase_ep_name = kNvTensorRTRTXExecutionProvider;
  std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  ProviderOptions provider_options;
  std::string key_prefix = "ep.";
  key_prefix += lowercase_ep_name;
  key_prefix += ".";

  for (const auto& [key, value] : config_options_map) {
    if (key.rfind(key_prefix, 0) == 0) {
      provider_options[key.substr(key_prefix.size())] = value;
    }
  }
  NvExecutionProviderInfo info = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(provider_options, config_options);

  auto ep = std::make_unique<NvExecutionProvider>(info);
  ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
  return ep;
}

struct Nv_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    NvExecutionProviderInfo info;
    info.device_id = device_id;

    return std::make_shared<NvProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    NvExecutionProviderInfo info = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(*provider_options, *config_options);
    return std::make_shared<NvProviderFactory>(info);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
