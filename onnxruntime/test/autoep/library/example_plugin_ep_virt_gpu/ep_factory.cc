// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#include "ep.h"
#include "../plugin_ep_utils.h"

EpFactoryVirtualGpu::EpFactoryVirtualGpu(const OrtApi& ort_api, const OrtEpApi& ep_api,
                                         const OrtModelEditorApi& model_editor_api,
                                         bool allow_virtual_devices,
                                         const OrtLogger& /*default_logger*/)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      model_editor_api_(model_editor_api),
      allow_virtual_devices_{allow_virtual_devices} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

EpFactoryVirtualGpu::~EpFactoryVirtualGpu() {
  if (virtual_hw_device_ != nullptr) {
    ep_api_.ReleaseHardwareDevice(virtual_hw_device_);
  }
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
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                     const OrtHardwareDevice* const* /*devices*/,
                                                                     size_t /*num_devices*/,
                                                                     OrtEpDevice** ep_devices,
                                                                     size_t max_ep_devices,
                                                                     size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<EpFactoryVirtualGpu*>(this_ptr);

  num_ep_devices = 0;

  // Create a virtual OrtHardwareDevice if application indicated it is allowed (e.g., for cross-compiling).
  // This example EP creates a virtual GPU OrtHardwareDevice and adds a new OrtEpDevice that uses the virtual GPU.
  if (factory->allow_virtual_devices_ && num_ep_devices < max_ep_devices) {
    OrtKeyValuePairs* hw_metadata = nullptr;
    factory->ort_api_.CreateKeyValuePairs(&hw_metadata);
    factory->ort_api_.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");

    auto* status = factory->ep_api_.CreateHardwareDevice(OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
                                                         factory->vendor_id_,
                                                         /*device_id*/ 0,
                                                         factory->vendor_.c_str(),
                                                         hw_metadata,
                                                         &factory->virtual_hw_device_);
    factory->ort_api_.ReleaseKeyValuePairs(hw_metadata);  // Release since ORT makes a copy.

    if (status != nullptr) {
      return status;
    }

    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
    factory->ort_api_.CreateKeyValuePairs(&ep_options);

    // made up example metadata values.
    factory->ort_api_.AddKeyValuePair(ep_metadata, "some_metadata", "1");
    factory->ort_api_.AddKeyValuePair(ep_options, "compile_optimization", "O3");

    OrtEpDevice* virtual_ep_device = nullptr;
    status = factory->ort_api_.GetEpApi()->CreateEpDevice(factory, factory->virtual_hw_device_, ep_metadata,
                                                          ep_options, &virtual_ep_device);

    factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
    factory->ort_api_.ReleaseKeyValuePairs(ep_options);

    if (status != nullptr) {
      return status;
    }

    ep_devices[num_ep_devices++] = virtual_ep_device;
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL EpFactoryVirtualGpu::CreateEpImpl(OrtEpFactory* this_ptr,
                                                          const OrtHardwareDevice* const* /*devices*/,
                                                          const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                          size_t num_devices,
                                                          const OrtSessionOptions* session_options,
                                                          const OrtLogger* logger,
                                                          OrtEp** ep) noexcept {
  auto* factory = static_cast<EpFactoryVirtualGpu*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for GPU and only expected to be selected for one GPU
    return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "EpFactoryVirtualGpu only supports selection for one device.");
  }

  std::string ep_context_enable;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, "ep.context_enable", "0", ep_context_enable));

  EpVirtualGpu::Config config = {};
  config.enable_ep_context = ep_context_enable == "1";

  auto actual_ep = std::make_unique<EpVirtualGpu>(*factory, config, *logger);

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
