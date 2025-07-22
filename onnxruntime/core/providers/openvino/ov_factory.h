// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "core/providers/shared_library/provider_api.h"
#include "openvino/openvino.hpp"

namespace onnxruntime {
namespace openvino_ep {

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

#define OVEP_DISABLE_MOVE(class_name) \
  class_name(class_name&&) = delete;  \
  class_name& operator=(class_name&&) = delete;

#define OVEP_DISABLE_COPY(class_name)     \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete;

#define OVEP_DISABLE_COPY_AND_MOVE(class_name) \
  OVEP_DISABLE_COPY(class_name)                \
  OVEP_DISABLE_MOVE(class_name)

template <typename Func>
static auto ApiEntry(Func&& func, std::optional<std::reference_wrapper<Ort::Logger>> logger = std::nullopt) {
  try {
    return func();
  } catch (const Ort::Exception& ex) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, ex.what());
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status(ex.what(), ex.GetOrtErrorCode()).release();
    }
  } catch (const std::exception& ex) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, ex.what());
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status(ex.what(), ORT_RUNTIME_EXCEPTION).release();
    }
  } catch (...) {
    if (logger) {
      ORT_CXX_LOG_NOEXCEPT(logger->get(), ORT_LOGGING_LEVEL_ERROR, "Unknown exception occurred.");
    }
    if constexpr (std::is_same_v<decltype(func()), OrtStatus*>) {
      return Ort::Status("Unknown exception occurred.", ORT_RUNTIME_EXCEPTION).release();
    }
  }
}

class OpenVINOEpPluginFactory : public OrtEpFactory, public ApiPtrs {
 public:
  OpenVINOEpPluginFactory(ApiPtrs apis, const std::string& ov_device, std::shared_ptr<ov::Core> ov_core);
  ~OpenVINOEpPluginFactory() = default;

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPluginFactory)

  static const std::vector<std::string>& GetOvDevices();

  std::vector<std::string> GetOvDevices(const std::string& device_type) {
    std::vector<std::string> filtered_devices;
    const auto& devices = GetOvDevices();
    std::copy_if(devices.begin(), devices.end(), std::back_inserter(filtered_devices),
                 [&device_type](const std::string& device) {
                   return device.find(device_type) != std::string::npos;
                 });
    return filtered_devices;
  }

  static const std::vector<std::string>& GetOvMetaDevices();

  // Member functions
  const char* GetName() const {
    return ep_name_.c_str();
  }

  const char* GetVendor() const {
    return vendor_;
  }

  OrtStatus* GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices);

  bool IsMetaDeviceFactory() const {
    return known_meta_devices_.find(device_type_) != known_meta_devices_.end();
  }

  // Constants
  static constexpr const char* vendor_ = "Intel";
  static constexpr uint32_t vendor_id_{0x8086};  // Intel's PCI vendor ID
  static constexpr const char* ov_device_key_ = "ov_device";
  static constexpr const char* ov_meta_device_key_ = "ov_meta_device";
  static constexpr const char* provider_name_ = "OpenVINOExecutionProvider";

 private:
  std::string ep_name_;
  std::string device_type_;
  std::vector<std::string> ov_devices_;
  std::shared_ptr<ov::Core> ov_core_;
  inline static const std::set<std::string> known_meta_devices_ = {
      "AUTO"};

 public:
  // Static callback methods for the OrtEpFactory interface
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const OpenVINOEpPluginFactory*>(this_ptr);
    return factory->GetName();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const OpenVINOEpPluginFactory*>(this_ptr);
    return factory->GetVendor();
  }

  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return OpenVINOEpPluginFactory::vendor_id_;
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept {
    auto* factory = static_cast<OpenVINOEpPluginFactory*>(this_ptr);
    return ApiEntry([&]() { return factory->GetSupportedDevices(devices, num_devices, ep_devices, max_ep_devices, p_num_ep_devices); });
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;  // return nullptr to indicate that this EP does not support data transfer.
    return nullptr;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory*) noexcept {
    return ORT_VERSION;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
