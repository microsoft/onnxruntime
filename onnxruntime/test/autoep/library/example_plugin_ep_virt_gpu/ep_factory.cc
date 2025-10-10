// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#include "ep.h"

EpFactoryVirtualGpu::EpFactoryVirtualGpu(const OrtApi& ort_api, const OrtEpApi& ep_api,
                                         const OrtLogger& default_logger)
    : OrtEpFactory{}, ort_api_(ort_api), ep_api_(ep_api), default_logger_{default_logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetAdditionalHardwareDevices = GetAdditionalHardwareDevicesImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

/*static*/
const char* ORT_API_CALL EpFactoryVirtualGpu::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const EpFactoryVirtualGpu*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL EpFactoryVirtualGpu::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const EpFactoryVirtualGpu*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL EpFactoryVirtualGpu::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const EpFactoryVirtualGpu*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL EpFactoryVirtualGpu::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const EpFactoryVirtualGpu*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::GetAdditionalHardwareDevicesImpl(OrtEpFactory* this_ptr,
                                                                              const OrtHardwareDevice* const* found_devices,
                                                                              size_t num_found_devices,
                                                                              OrtHardwareDevice** additional_devices,
                                                                              size_t max_additional_devices,
                                                                              size_t* num_additional_devices) noexcept {
  // EP factory can provide ORT with additional hardware devices that ORT did not find, or more likely, that are not
  // available on the target machine but could serve as compilation targets.

  // As an example, this example EP factory will first look for a GPU device among the devices found by ORT. If there
  // is no GPU available, then this EP will create a virtual GPU device that the application can use a compilation target.

  auto* factory = static_cast<EpFactoryVirtualGpu*>(this_ptr);
  bool found_gpu = false;

  for (size_t i = 0; i < num_found_devices; ++i) {
    const OrtHardwareDevice& device = *found_devices[i];

    if (factory->ort_api_.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        factory->ort_api_.HardwareDevice_Vendor(&device) == factory->vendor_) {
      found_gpu = true;
      break;
    }
  }

  *num_additional_devices = 0;

  if (!found_gpu && max_additional_devices >= 1) {
    // Create a new HW device.
    OrtKeyValuePairs* hw_metadata = nullptr;
    factory->ort_api_.CreateKeyValuePairs(&hw_metadata);
    factory->ort_api_.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_DiscoveredBy,
                                      factory->ep_name_.c_str());
    factory->ort_api_.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");

    OrtHardwareDevice* new_device = nullptr;
    auto* status = factory->ep_api_.CreateHardwareDevice(OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
                                                         factory->vendor_id_,
                                                         /*device_id*/ 0,
                                                         factory->vendor_.c_str(),
                                                         hw_metadata,
                                                         &new_device);
    factory->ort_api_.ReleaseKeyValuePairs(hw_metadata);  // Release since ORT makes a copy.

    if (status != nullptr) {
      return status;
    }

    // ORT will take ownership of the new HW device.
    additional_devices[0] = new_device;
    *num_additional_devices = 1;
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                     const OrtHardwareDevice* const* devices,
                                                                     size_t num_devices,
                                                                     OrtEpDevice** ep_devices,
                                                                     size_t max_ep_devices,
                                                                     size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<EpFactoryVirtualGpu*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api_.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        factory->ort_api_.HardwareDevice_Vendor(&device) == factory->vendor_) {
      // these can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);

      // random example using made up values
      factory->ort_api_.AddKeyValuePair(ep_metadata, "ex_key", "ex_value");
      factory->ort_api_.AddKeyValuePair(ep_options, "compile_optimization", "O3");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api_.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                  &ep_device);

      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::CreateEpImpl(OrtEpFactory* this_ptr,
                                                          const OrtHardwareDevice* const* /*devices*/,
                                                          const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                          size_t num_devices,
                                                          const OrtSessionOptions* /*session_options*/,
                                                          const OrtLogger* logger,
                                                          OrtEp** ep) noexcept {
  auto* factory = static_cast<EpFactoryVirtualGpu*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for GPU and only expected to be selected for one GPU
    return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "EpFactoryVirtualGpu only supports selection for one device.");
  }

  auto actual_ep = std::make_unique<EpVirtualGpu>(*factory, *logger);

  *ep = actual_ep.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL EpFactoryVirtualGpu::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  EpVirtualGpu* dummy_ep = static_cast<EpVirtualGpu*>(ep);
  delete dummy_ep;
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::CreateAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                                 const OrtMemoryInfo* /*memory_info*/,
                                                                 const OrtKeyValuePairs* /*allocator_options*/,
                                                                 OrtAllocator** allocator) noexcept {
  // Don't support custom allocators in this example for simplicity. A GPU EP would normally support allocators.
  *allocator = nullptr;
  return nullptr;
}

/*static*/
void ORT_API_CALL EpFactoryVirtualGpu::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                            OrtAllocator* /*allocator*/) noexcept {
  // Do nothing.
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                                    OrtDataTransferImpl** data_transfer) noexcept {
  // Don't support data transfer in this example for simplicity. A GPU EP would normally support it.
  *data_transfer = nullptr;
  return nullptr;
}

/*static*/
bool ORT_API_CALL EpFactoryVirtualGpu::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::CreateSyncStreamForDeviceImpl(OrtEpFactory* /*this_ptr*/,
                                                                           const OrtMemoryDevice* /*memory_device*/,
                                                                           const OrtKeyValuePairs* /*stream_options*/,
                                                                           OrtSyncStreamImpl** stream) noexcept {
  // Don't support sync streams in this example. A GPU EP would normally support it.
  *stream = nullptr;
  return nullptr;
}
