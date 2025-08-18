// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_DML)

#include "core/session/plugin_ep/ep_factory_dml.h"

#include "core/framework/error_code_helper.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

OrtStatus* DmlEpFactory::GetSupportedDevices(EpFactoryInternal& ep_factory,
                                             const OrtHardwareDevice* const* devices,
                                             size_t num_devices,
                                             OrtEpDevice** ep_devices,
                                             size_t max_ep_devices,
                                             size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      auto ep_options = std::make_unique<OrtKeyValuePairs>();

      // TODO: Should we ignore a user provided 'device_id' when they select an OrtEpDevice as that is
      //       associated with a specific device.
      //       How would we know what options should not allow user overrides if set in OrtEpDevice?
      int32_t device_id = 0;  // If no device_id was found default to 0
      if (auto it = device.metadata.Entries().find("DxgiAdapterNumber"); it != device.metadata.Entries().end()) {
        device_id = std::stoi(it->second);
      }

      ep_options->Add("device_id", std::to_string(device_id));

      auto* api_status = OrtExecutionProviderApi::CreateEpDevice(&ep_factory,
                                                                 &device, nullptr, ep_options.get(),
                                                                 &ep_devices[num_ep_devices]);

      if (device_memory_infos.size() < device_id + 1) {
        device_memory_infos.resize(device_id + 1);
        device_allocators.resize(device_id + 1);
      }

      if (device_memory_infos[device_id] == nullptr) {
        // Create memory info for the device if it doesn't already exist
        device_memory_infos[device_id] = std::make_unique<OrtMemoryInfo>(
            "DML", OrtAllocatorType::OrtDeviceAllocator,
            OrtDevice(OrtDevice::DML, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::MICROSOFT,
                      narrow<int16_t>(device_id)));
      }

      // This is what we need to add once CreateAllocator is implemented to create a shared allocator for the device.
      // OrtExecutionProviderApi::EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices],
      //                                                   device_memory_infos[device_id].get());

      if (api_status != nullptr) {
        return api_status;
      }

      ++num_ep_devices;
    }
  }

  return nullptr;
}

OrtStatus* DmlEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                                  const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                                  size_t num_devices,
                                                  const OrtSessionOptions* session_options,
                                                  const OrtLogger* session_logger,
                                                  std::unique_ptr<IExecutionProvider>* ep) noexcept {
  *ep = nullptr;

  if (num_devices != 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "DML EP factory currently only supports one device at a time.");
  }

  auto ep_options = GetOptionsFromSessionOptions(session_options->value);
  auto dml_ep_factory = DMLProviderFactoryCreator::CreateFromProviderOptions(session_options->value.config_options,
                                                                             ep_options);

  *ep = dml_ep_factory->CreateProvider();
  (*ep)->SetLogger(session_logger->ToInternal());

  return nullptr;
}

/*
// TODO: This needs to create an allocator for the specific device so it's available as a shared allocator. That
//       requires pulling lots of things out of the DML EP to get the D3D12 device and create a
//       BucketizedBufferAllocator. See providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp
OrtStatus* DmlEpFactory::CreateAllocator(const OrtMemoryInfo* memory_info,
                                         const OrtKeyValuePairs* allocator_options,
                                         OrtAllocator** allocator) noexcept {
}

// TODO: Wrap the IDataTransfer implementation so we can copy to device using OrtApi CopyTensors.
OrtStatus* DmlEpFactory::CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept {
}
*/
}  // namespace onnxruntime

#endif  // USE_DML
