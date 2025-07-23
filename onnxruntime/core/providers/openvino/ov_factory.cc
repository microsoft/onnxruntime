// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <vector>
#include <ranges>
#include <format>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_c_api.h"
#include "ov_factory.h"
#include "openvino/openvino.hpp"
#include "ov_interface.h"

using namespace onnxruntime::openvino_ep;
using ov_core_singleton = onnxruntime::openvino_ep::WeakSingleton<ov::Core>;

static void InitCxxApi(const OrtApiBase& ort_api_base) {
  static std::once_flag init_api;
  std::call_once(init_api, [&]() {
    const OrtApi* ort_api = ort_api_base.GetApi(ORT_API_VERSION);
    Ort::InitApi(ort_api);
  });
}

OpenVINOEpPluginFactory::OpenVINOEpPluginFactory(ApiPtrs apis, const std::string& ov_metadevice_name, std::shared_ptr<ov::Core> core)
    : ApiPtrs{apis},
      ep_name_(ov_metadevice_name.empty() ? provider_name_ : std::string(provider_name_) + "." + ov_metadevice_name),
      device_type_(ov_metadevice_name),
      ov_core_(std::move(core)) {
  OrtEpFactory::GetName = GetNameImpl;
  OrtEpFactory::GetVendor = GetVendorImpl;
  OrtEpFactory::GetVendorId = GetVendorIdImpl;
  OrtEpFactory::GetSupportedDevices = GetSupportedDevicesImpl;
  OrtEpFactory::GetVersion = GetVersionImpl;
  OrtEpFactory::CreateDataTransfer = CreateDataTransferImpl;

  ort_version_supported = ORT_API_VERSION;  // Set to the ORT version we were compiled with.
}

const std::vector<std::string>& OpenVINOEpPluginFactory::GetOvDevices() {
  static std::vector<std::string> devices = ov_core_singleton::Get()->get_available_devices();
  return devices;
}

const std::vector<std::string>& OpenVINOEpPluginFactory::GetOvMetaDevices() {
  static std::vector<std::string> virtual_devices = [ov_core = ov_core_singleton::Get()] {
    std::vector<std::string> supported_virtual_devices{};
    for (const auto& meta_device : known_meta_devices_) {
      try {
        ov_core->get_property(meta_device, ov::supported_properties);
        supported_virtual_devices.push_back(meta_device);
      } catch (ov::Exception&) {
        // meta device isn't supported.
      }
    }
    return supported_virtual_devices;
  }();

  return virtual_devices;
}

OrtStatus* OpenVINOEpPluginFactory::GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                                        size_t num_devices,
                                                        OrtEpDevice** ep_devices,
                                                        size_t max_ep_devices,
                                                        size_t* p_num_ep_devices) {
  size_t& num_ep_devices = *p_num_ep_devices;

  // Create a map for device type mapping
  static const std::map<OrtHardwareDeviceType, std::string> ort_to_ov_device_name = {
      {OrtHardwareDeviceType::OrtHardwareDeviceType_CPU, "CPU"},
      {OrtHardwareDeviceType::OrtHardwareDeviceType_GPU, "GPU"},
      {OrtHardwareDeviceType::OrtHardwareDeviceType_NPU, "NPU"},
  };

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (ort_api.HardwareDevice_VendorId(&device) != vendor_id_) {
      // Not an Intel Device.
      continue;
    }

    auto device_type = ort_api.HardwareDevice_Type(&device);
    auto device_it = ort_to_ov_device_name.find(device_type);
    if (device_it == ort_to_ov_device_name.end()) {
      // We don't know about this device type
      continue;
    }

    const auto& ov_device_type = device_it->second;
    std::string ov_device_name;
    auto get_pci_device_id = [&](const std::string& ov_device) {
      try {
        ov::device::PCIInfo pci_info = ov_core_->get_property(ov_device, ov::device::pci_info);
        return pci_info.device;
      } catch (ov::Exception&) {
        return 0u;  // If we can't get the PCI info, we won't have a device ID.
      }
    };

    auto filtered_devices = GetOvDevices(ov_device_type);
    auto matched_device = filtered_devices.begin();
    if (filtered_devices.size() > 1 && device_type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // If there are multiple devices of the same type, we need to match by device ID.
      matched_device = std::find_if(filtered_devices.begin(), filtered_devices.end(), [&](const std::string& ov_device) {
        uint32_t ort_device_id = ort_api.HardwareDevice_DeviceId(&device);
        return ort_device_id == get_pci_device_id(ov_device);
      });
    }

    if (matched_device == filtered_devices.end()) {
      // We didn't find a matching OpenVINO device for the OrtHardwareDevice.
      continue;
    }

    // these can be returned as nullptr if you have nothing to add.
    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    ort_api.CreateKeyValuePairs(&ep_metadata);
    ort_api.AddKeyValuePair(ep_metadata, ov_device_key_, matched_device->c_str());

    if (IsMetaDeviceFactory()) {
      ort_api.AddKeyValuePair(ep_metadata, ov_meta_device_key_, device_type_.c_str());
    }

    // Create EP device
    auto* status = ort_api.GetEpApi()->CreateEpDevice(this, &device, ep_metadata, ep_options,
                                                      &ep_devices[num_ep_devices++]);

    ort_api.ReleaseKeyValuePairs(ep_metadata);
    ort_api.ReleaseKeyValuePairs(ep_options);

    if (status != nullptr) {
      return status;
    }
  }

  return nullptr;
}

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* /*default_logger*/,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  InitCxxApi(*ort_api_base);
  const ApiPtrs api_ptrs{Ort::GetApi(), Ort::GetEpApi(), Ort::GetModelEditorApi()};

  // Get available devices from OpenVINO
  auto ov_core = ov_core_singleton::Get();
  std::vector<std::string> supported_factories = {""};
  const auto& meta_devices = OpenVINOEpPluginFactory::GetOvMetaDevices();
  supported_factories.insert(supported_factories.end(), meta_devices.begin(), meta_devices.end());

  const size_t required_factories = supported_factories.size();
  if (max_factories < required_factories) {
    return Ort::Status(std::format("Not enough space to return EP factories. Need at least {} factories.", required_factories).c_str(), ORT_INVALID_ARGUMENT);
  }

  size_t factory_index = 0;
  for (const auto& device_name : supported_factories) {
    // Create a factory for this specific device
    factories[factory_index++] = new OpenVINOEpPluginFactory(api_ptrs, device_name, ov_core);
  }

  *num_factories = factory_index;
  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<OpenVINOEpPluginFactory*>(factory);
  return nullptr;
}
}
