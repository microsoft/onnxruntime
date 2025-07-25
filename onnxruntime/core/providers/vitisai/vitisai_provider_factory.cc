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

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t /*num_devices*/,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    auto ep_factory = CreateExecutionProviderFactory(&provider_options);
    ep = ep_factory->CreateProvider(session_options, logger);
    return Status::OK();
  }
} g_provider;

struct VitisAIEpFactory : OrtEpFactory {
  VitisAIEpFactory(const OrtApi& ort_api_in, const OrtLogger& default_logger_in)
      : ort_api{ort_api_in}, default_logger{default_logger_in} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;

    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;

    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  }

  static const char* GetNameImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ep_name;
  }

  static const char* GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return vendor;
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return hardware_vendor_id;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* ep_factory,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    VitisAIEpFactory* factory = static_cast<VitisAIEpFactory*>(ep_factory);

    for (size_t i = 0; i < num_devices; ++i) {
      const OrtHardwareDevice* hardware_device = devices[i];
      const std::uint32_t vendor_id = factory->ort_api.HardwareDevice_VendorId(hardware_device);
      const OrtHardwareDeviceType device_type = factory->ort_api.HardwareDevice_Type(hardware_device);

      if ((vendor_id != VitisAIEpFactory::hardware_vendor_id) ||
          (device_type != OrtHardwareDeviceType_NPU)) {
        continue;
      }

      if (num_ep_devices == max_ep_devices) {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Not enough space to return EP devices.");
      }

      auto status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, hardware_device, nullptr, nullptr,
                                                                &ep_devices[num_ep_devices++]);
      if (status != nullptr) {
        return status;
      }
    }
    return nullptr;
  }

  static OrtStatus* CreateDataTransferImpl(
      OrtEpFactory* /*this_ptr*/,
      OrtDataTransferImpl** /*data_transfer*/) noexcept {
    return nullptr;  // TODO: Implement data transfer if needed
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return CreateStatus(ORT_INVALID_ARGUMENT, "VitisAI EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory*, OrtEp*) noexcept {
    // no-op as we never create an EP here.
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* /*memory_info*/,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept {
    auto* factory = static_cast<VitisAIEpFactory*>(this_ptr);

    *allocator = nullptr;
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CreateAllocator should not be called as we did not add OrtMemoryInfo to our OrtEpDevice.");
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
    // should never be called as we don't implement CreateAllocator
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;  // not implemented
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* /*memory_device*/,
                                                               const OrtKeyValuePairs* /*stream_options*/,
                                                               OrtSyncStreamImpl** stream) noexcept {
    auto* factory = static_cast<VitisAIEpFactory*>(this_ptr);

    *stream = nullptr;
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT, "CreateSyncStreamForDevice should not be called as IsStreamAware returned false.");
  }

  const OrtApi& ort_api;
  const OrtLogger& default_logger;
  static constexpr const char* const ep_name{kVitisAIExecutionProvider};
  static constexpr std::uint32_t hardware_vendor_id{0x1022};
  static constexpr const char* const vendor{"AMD"};
};

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}

OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }
  factories[0] = std::make_unique<onnxruntime::VitisAIEpFactory>(*ort_api, *default_logger).release();
  *num_factories = 1;
  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<onnxruntime::VitisAIEpFactory*>(factory);
  return nullptr;
}
}
