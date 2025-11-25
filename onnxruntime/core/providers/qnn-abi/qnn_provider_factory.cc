// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn-abi/qnn_provider_factory.h"

#include <cassert>
#include <iostream>

#include "core/framework/error_code_helper.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/qnn_allocator.h"

namespace onnxruntime {

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
QnnEpFactory::QnnEpFactory(const char* ep_name,
                           ApiPtrs ort_api_in,
                           std::unordered_map<OrtHardwareDeviceType, std::string> supported_backends)
    : OrtEpFactory{}, ApiPtrs(ort_api_in), ep_name_{ep_name}, supported_backends_{supported_backends} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;

  // HOST_ACCESSIBLE memory.
  OrtMemoryInfo* mem_info = nullptr;
  auto* status = ort_api.CreateMemoryInfo_V2("QnnHtpShared",
                                             OrtMemoryInfoDeviceType_CPU,
                                             /*vendor*/ 0x5143,
                                             /*device_id*/ 0,
                                             OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                             /*alignment*/ 0,
                                             OrtAllocatorType::OrtDeviceAllocator,
                                             &mem_info);
  if (status != nullptr) {
    ort_api.ReleaseMemoryInfo(mem_info);
  }
  host_accessible_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);
}

// Returns the name for the EP. Each unique factory configuration must have a unique name.
// Ex: a factory that supports NPU should have a different than a factory that supports GPU.
const char* ORT_API_CALL QnnEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL QnnEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

uint32_t ORT_API_CALL QnnEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

const char* ORT_API_CALL QnnEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

// Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
// An EP created with this factory is expected to be able to execute a model with *all* supported
// hardware devices at once. A single instance of QNN EP is not currently setup to partition a model among
// multiple different QNN backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
// support one backend: npu. To support a different backend, like gpu, create a different factory instance
// that only supports GPU.
OrtStatus* ORT_API_CALL QnnEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                              const OrtHardwareDevice* const* devices,
                                                              size_t num_devices,
                                                              OrtEpDevice** ep_devices,
                                                              size_t max_ep_devices,
                                                              size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);

  for (size_t idx = 0; idx < num_devices && num_ep_devices < max_ep_devices; ++idx) {
    const OrtHardwareDevice& device = *devices[idx];
    auto device_type = factory->ort_api.HardwareDevice_Type(&device);
    auto vendor_id = factory->ort_api.HardwareDevice_VendorId(&device);
    auto supported_backend_it = factory->supported_backends_.find(device_type);

    if (vendor_id == factory->vendor_id_ || device_type == OrtHardwareDeviceType_CPU) {
      OrtEpDevice* ep_device = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;

      // This option is set for auto EP select usage where `backend_type` or `backend_path` may not be given.
      // The key is deliberately prefixed with `ep_select_` to avoid conflict with existing `backend_path`.
      // Note that since HTP backend can be run on CPU through emulation, we could not determine which backend library
      // to be used. Such case is skipped to set this option and relied on user-provided one.
      if (supported_backend_it != factory->supported_backends_.end() && device_type != OrtHardwareDeviceType_CPU) {
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.AddKeyValuePair(ep_options, "ep_select_backend_path", supported_backend_it->second.c_str());
      }

      OrtStatus* status = factory->ep_api.CreateEpDevice(factory, &device, nullptr, ep_options, &ep_device);
      ep_devices[num_ep_devices++] = ep_device;
      factory->ep_devices_.push_back(ep_device);

      if (ep_options != nullptr) {
        factory->ort_api.ReleaseKeyValuePairs(ep_options);
      }
      RETURN_IF_NOT_NULL(status);
    }
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                   _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                                   _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                   _In_ size_t /*num_devices*/,  // Mark as unused
                                                   _In_ const OrtSessionOptions* session_options,
                                                   _In_ const OrtLogger* logger,
                                                   _Out_ OrtEp** ep) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  *ep = nullptr;

  // Create the execution provider
  RETURN_IF_NOT_NULL(factory->ort_api.Logger_LogMessage(logger,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        "Creating QNN EP", ORT_FILE, __LINE__, __FUNCTION__));

  std::unique_ptr<QnnEp> qnn_ep;
  try {
    qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options, logger);

    // Setting allocator info is delayed from GetSupportedDevices to here as QNN-EP relies on provider options to
    // determine whether to use HTP shared memory but they are not available until now. This workaround works since
    // PluginExecutionProvider collects the allocator infos after creating the EP (refer to
    // ep_plugin_provider_interfaces.cc for the detail flow).
    std::string enable_htp_shared_memory_allocator_str;
    GetSessionConfigEntryOrDefault(factory->ort_api,
                                   *session_options,
                                   GetProviderOptionPrefix(factory->ep_name_) + "enable_htp_shared_memory_allocator",
                                   "0",
                                   enable_htp_shared_memory_allocator_str);
    if (enable_htp_shared_memory_allocator_str == "1") {
      for (OrtEpDevice* ep_device : factory->ep_devices_) {
        RETURN_IF_NOT_NULL(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->host_accessible_memory_info_.get()));
      }
    }
  } catch (const std::runtime_error& e) {
    return factory->ort_api.CreateStatus(ORT_FAIL, e.what());
  }

  *ep = qnn_ep.release();
  return nullptr;
}

void ORT_API_CALL QnnEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  if (ep == nullptr) {
    return;
  }

  QnnEp* dummy_ep = static_cast<QnnEp*>(ep);
  delete dummy_ep;
}

void ORT_API_CALL QnnEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  delete static_cast<qnn::HtpSharedMemoryAllocator*>(allocator);
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateDataTransferImpl(OrtEpFactory* /* this_ptr */,
                                                             OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;

  return nullptr;
}

bool ORT_API_CALL QnnEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

}  // namespace onnxruntime

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories) {
  if (ort_api_base == nullptr) {
    return nullptr;  // Cannot create status without API base
  }

  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    return nullptr;  // Cannot create status without ORT API
  }

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  // We allow `backend_type` (e.g., `htp`) or `backend_path` in relateive path (e.g., `QnnHtp.dll`) for configurations,
  // and QnnBackendManager will later find the appropriate library and load it relative to the OnnxRuntime library.
  // But if QNN-EP is distributed separately from the OnnxRuntime library (i.e., EP ABI), the backend library may well
  // not be relative to the OnnxRuntime but to the EP library itself instead.
  // If EP library is co-located with the OnnxRuntime library, then this is consistent with the existing behavior, but
  // a EP library that is shipped 'out-of-band' will use a backend relative to itself.
  std::unordered_map<OrtHardwareDeviceType, std::string> supported_backends = {
#if defined(_WIN32)
      {OrtHardwareDeviceType_NPU, "QnnHtp.dll"},
      {OrtHardwareDeviceType_GPU, "QnnGpu.dll"},
#else
      {OrtHardwareDeviceType_NPU, "libQnnHtp.so"},
      {OrtHardwareDeviceType_GPU, "libQnnGpu.so"},
#endif
  };

  for (auto& [_, backend_path] : supported_backends) {
    // Identify the path of the current dynamic library, and expect that backend_path is in the same directory.
    std::basic_string<ORTCHAR_T> current_path = onnxruntime::GetDynamicLibraryLocationByAddress(
        reinterpret_cast<const void*>(&CreateEpFactories));

    if (!current_path.empty()) {
      const std::filesystem::path parent_path = std::filesystem::path{std::move(current_path)}.parent_path();
      backend_path = (parent_path / backend_path).string();
    }
  }

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  if (factories == nullptr || num_factories == nullptr) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid arguments: factories and num_factories cannot be null.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get EP API.");
  }

  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
  if (model_editor_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get Model Editor API.");
  }

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<onnxruntime::QnnEpFactory> factory;
  try {
    factory = std::make_unique<onnxruntime::QnnEpFactory>(registration_name,
                                                          onnxruntime::ApiPtrs{*ort_api,
                                                                               *ep_api,
                                                                               *model_editor_api},
                                                          supported_backends);
  } catch (const std::exception& e) {
    return ort_api->CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api->CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP factory.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  // Set default logger for later use.
  onnxruntime::OrtLoggingManager::SetDefaultLogger(default_logger);

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  if (factory == nullptr) {
    return nullptr;
  }

  delete static_cast<onnxruntime::QnnEpFactory*>(factory);
  return nullptr;
}
}
