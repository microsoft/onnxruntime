// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <string>
#include <unordered_map>
#include "core/providers/qnn/qnn_provider_factory_creator.h"
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map, const ConfigOptions* config_options)
      : provider_options_map_(provider_options_map), config_options_(config_options) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_, config_options_);
  }

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // We extract those EP options and pass them to QNN EP as separate "provider options".
    std::unordered_map<std::string, std::string> provider_options = provider_options_map_;
    std::string key_prefix = "ep.";
    key_prefix += qnn::utils::GetLowercaseString(kQnnExecutionProvider);
    key_prefix += ".";

    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    auto qnn_ep = std::make_unique<QNNExecutionProvider>(provider_options, &config_options);
    qnn_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return qnn_ep;
  }

 private:
  ProviderOptions provider_options_map_;
  const ConfigOptions* config_options_;
};

#if BUILD_QNN_EP_STATIC_LIB
std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                                                             const SessionOptions* session_options) {
  const ConfigOptions* config_options = nullptr;
  if (session_options != nullptr) {
    config_options = &session_options->config_options;
  }

  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map, config_options);
}
#else
struct QNN_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) override {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    return std::make_shared<onnxruntime::QNNProviderFactory>(*provider_options, config_options);
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (num_devices != 1) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "QNN EP only supports one device.");
    }

    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);
    auto ep_factory = CreateExecutionProviderFactory(arg);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
  }

  void Initialize() override {}
  void Shutdown() override {}
} g_provider;
#endif  // BUILD_QNN_EP_STATIC_LIB

}  // namespace onnxruntime

#if !BUILD_QNN_EP_STATIC_LIB
extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

#include "core/framework/error_code_helper.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
struct QnnEpFactory : OrtEpFactory {
  QnnEpFactory(const OrtApi& ort_api_in,
               const char* ep_name,
               OrtHardwareDeviceType hw_type,
               const char* qnn_backend_type)
      : ort_api{ort_api_in}, ep_name{ep_name}, ort_hw_device_type{hw_type}, qnn_backend_type{qnn_backend_type} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_vendor_id;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of QNN EP is not currently setup to partition a model among
  // multiple different QNN backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
  // support one backend: npu. To support a different backend, like gpu, create a different factory instance
  // that only supports GPU.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == factory->ort_hw_device_type &&
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.AddKeyValuePair(ep_options, "backend_type", factory->qnn_backend_type.c_str());
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
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "QNN EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  const OrtApi& ort_api;
  const std::string ep_name;                 // EP name
  const std::string ep_vendor{"Microsoft"};  // EP vendor name
  uint32_t ep_vendor_id{0x1414};             // Microsoft vendor ID

  // Qualcomm vendor ID. Refer to the ACPI ID registry (search Qualcomm): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  const OrtHardwareDeviceType ort_hw_device_type;  // Supported OrtHardwareDevice
  const std::string qnn_backend_type;              // QNN backend type for OrtHardwareDevice
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  // Factory could use registration_name or define its own EP name.
  auto factory_npu = std::make_unique<QnnEpFactory>(*ort_api,
                                                    onnxruntime::kQnnExecutionProvider,
                                                    OrtHardwareDeviceType_NPU, "htp");

  // If want to support GPU, create a new factory instance because QNN EP is not currently setup to partition a single model
  // among heterogeneous devices.
  // std::unique_ptr<OrtEpFactory> factory_gpu = std::make_unique<QnnEpFactory>(*ort_api, "QNNExecutionProvider_GPU", OrtHardwareDeviceType_GPU, "gpu");

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory_npu.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<QnnEpFactory*>(factory);
  return nullptr;
}
}
#endif  // !BUILD_QNN_EP_STATIC_LIB
