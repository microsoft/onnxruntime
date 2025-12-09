// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#include "ep.h"
#include "ep_kernel_registration.h"
#include "../plugin_ep_utils.h"

ExampleKernelEpFactory::ExampleKernelEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                                               const OrtLogger& /*default_logger*/)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api) {
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

ExampleKernelEpFactory::~ExampleKernelEpFactory() {
  if (kernel_registry_ != nullptr) {
    Ort::GetEpApi().ReleaseKernelRegistry(kernel_registry_);
  }
}

OrtStatus* ExampleKernelEpFactory::GetKernelRegistryForEp(ExampleKernelEp& ep,
                                                          const OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  if (kernel_registry_ == nullptr) {
    void* op_kernel_state = nullptr;  // Optional state that is provided to kernels on creation (can be null).
    const char* ep_name = ep.GetName(static_cast<const OrtEp*>(&ep));

    // This statement creates the kernel registry and caches it in the OrtEpFactory instance.
    // We assume that all EPs created by this factory can use the same kernel registry. This may not be the
    // case in a more complex OrtEpFactory that can create EP instances that are each configured for different
    // hardware devices. In such a scenario, a different kernel registry may be created for each EP configuration.
    RETURN_IF_ERROR(CreateKernelRegistry(ep_name, op_kernel_state, &kernel_registry_));
  }

  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

/*static*/
const char* ORT_API_CALL ExampleKernelEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleKernelEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL ExampleKernelEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleKernelEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL ExampleKernelEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleKernelEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL ExampleKernelEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleKernelEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                        const OrtHardwareDevice* const* hw_devices,
                                                                        size_t num_devices,
                                                                        OrtEpDevice** ep_devices,
                                                                        size_t max_ep_devices,
                                                                        size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<ExampleKernelEpFactory*>(this_ptr);

  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *hw_devices[i];
    if (factory->ort_api_.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);

      // random example using made up values
      factory->ort_api_.AddKeyValuePair(ep_metadata, "supported_devices", "CrackGriffin 7+");
      factory->ort_api_.AddKeyValuePair(ep_options, "run_really_fast", "true");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api_.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                  &ep_device);

      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      // register the allocator info required by the EP.
      // registering OrtMemoryInfo for host accessible memory would be done in an additional call.
      // OrtReadOnlyAllocator + OrtDeviceMemoryType_DEFAULT allocator for use with initializers is optional.
      // RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_.get()));
      // RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_.get()));

      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                             const OrtHardwareDevice* const* /*devices*/,
                                                             const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                             size_t num_devices,
                                                             const OrtSessionOptions* /*session_options*/,
                                                             const OrtLogger* logger,
                                                             OrtEp** ep) noexcept {
  auto* factory = static_cast<ExampleKernelEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->ort_api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "ExampleKernelEpFactory only supports selection for one device.");
  }

  auto actual_ep = std::make_unique<ExampleKernelEp>(*factory, *logger);
  *ep = actual_ep.release();

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleKernelEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<ExampleKernelEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpFactory::CreateAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                                    const OrtMemoryInfo* /*memory_info*/,
                                                                    const OrtKeyValuePairs* /*allocator_options*/,
                                                                    OrtAllocator** allocator) noexcept {
  // Don't support custom allocators in this example for simplicity. A GPU EP would normally support allocators.
  *allocator = nullptr;
  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleKernelEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                               OrtAllocator* /*allocator*/) noexcept {
  // Do nothing.
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpFactory::CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                                       OrtDataTransferImpl** data_transfer) noexcept {
  // Don't support data transfer in this example for simplicity. A GPU EP would normally support it.
  *data_transfer = nullptr;
  return nullptr;
}

/*static*/
bool ORT_API_CALL ExampleKernelEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* /*this_ptr*/,
                                                                              const OrtMemoryDevice* /*memory_device*/,
                                                                              const OrtKeyValuePairs* /*stream_opts*/,
                                                                              OrtSyncStreamImpl** stream) noexcept {
  // Don't support sync streams in this example.
  *stream = nullptr;
  return nullptr;
}
