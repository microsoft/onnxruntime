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

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t /*num_devices*/,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);
    auto ep_factory = CreateExecutionProviderFactory(arg);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
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

#include "core/framework/error_code_helper.h"

// OrtEpApi infrastructure to be able to use the NvTensorRTRTX EP as an OrtEpFactory for auto EP selection.
struct NvTensorRtRtxEpFactory : OrtEpFactory {
  NvTensorRtRtxEpFactory(const OrtApi& ort_api_in,
                         const OrtLogger& default_logger_in,
                         const char* ep_name,
                         OrtHardwareDeviceType hw_type)
      : ort_api{ort_api_in}, default_logger{default_logger_in}, ep_name{ep_name}, ort_hw_device_type{hw_type} {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVersion = GetVersionImpl;
    GetVendorId = GetVendorIdImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->vendor_id;
  }


  static const char* GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of NvTensorRtRtx EP is not currently setup to partition a model among
  // multiple different NvTensorRtRtx backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
  // support one backend: gpu. To support a different backend, like npu, create a different factory instance
  // that only supports NPU.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<NvTensorRtRtxEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == factory->ort_hw_device_type &&
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        ORT_API_RETURN_IF_ERROR(
            factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, ep_options,
                                                        &ep_devices[num_ep_devices++]));
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  const OrtApi& ort_api;
  const OrtLogger& default_logger;
  const std::string ep_name;
  const std::string vendor{"NVIDIA"};

  // NVIDIA vendor ID. Refer to the ACPI ID registry (search NVIDIA): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id{0x10de};
  const OrtHardwareDeviceType ort_hw_device_type;  // Supported OrtHardwareDevice
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  // Factory could use registration_name or define its own EP name.
  auto factory_gpu = std::make_unique<NvTensorRtRtxEpFactory>(*ort_api, *default_logger,
                                                              onnxruntime::kNvTensorRTRTXExecutionProvider,
                                                              OrtHardwareDeviceType_GPU);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory_gpu.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<NvTensorRtRtxEpFactory*>(factory);
  return nullptr;
}
}
