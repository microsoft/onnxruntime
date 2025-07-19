// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "ep.h"
#include "ep_allocator.h"
#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "ep_stream_support.h"

ExampleEpFactory::ExampleEpFactory(const char* ep_name, ApiPtrs apis)
    : ApiPtrs(apis), ep_name_{ep_name} {
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

  // setup the OrtMemoryInfo instances required by the EP.
  // We pretend the device the EP is running on is GPU.
  OrtMemoryInfo* mem_info = nullptr;
  auto* status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU", OrtMemoryInfoDeviceType_GPU,
                                             /*vendor*/ 0xBE57, /* device_id */ 0,
                                             OrtDeviceMemoryType_DEFAULT,
                                             /*alignment*/ 0,
                                             // it is invalid to use OrtArenaAllocator as that is reserved for the
                                             // internal ORT Arena implementation
                                             OrtAllocatorType::OrtDeviceAllocator,
                                             &mem_info);
  assert(status == nullptr);  // should never fail.
  default_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

  // create data transfer for the device
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_.get());
  data_transfer_impl_ = std::make_unique<ExampleDataTransfer>(apis, device);

  // HOST_ACCESSIBLE memory example. use the non-CPU device type so it's clear which device the memory is also
  // accessible from. we infer from the type of HOST_ACCESSIBLE that it's CPU accessible.
  mem_info = nullptr;
  status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU pinned", OrtMemoryInfoDeviceType_GPU,
                                       /*vendor*/ 0xBE57, /* device_id */ 0,
                                       OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                       /*alignment*/ 0,
                                       OrtAllocatorType::OrtDeviceAllocator,
                                       &mem_info);
  ort_api.ReleaseMemoryInfo(mem_info);
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL ExampleEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL ExampleEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                  const OrtHardwareDevice* const* devices,
                                                                  size_t num_devices,
                                                                  OrtEpDevice** ep_devices,
                                                                  size_t max_ep_devices,
                                                                  size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api.CreateKeyValuePairs(&ep_options);

      // random example using made up values
      factory->ort_api.AddKeyValuePair(ep_metadata, "supported_devices", "CrackGriffin 7+");
      factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

      // OrtEpDevice copies ep_metadata and ep_options.
      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                 &ep_device);

      factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      // register the allocator info required by the EP.
      // registering OrtMemoryInfo for host accessible memory would be done in an additional call.
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_.get()));

      ep_devices[num_ep_devices++] = ep_device;
    }

    // C++ API equivalent. Throws on error.
    //{
    //  Ort::ConstHardwareDevice device(devices[i]);
    //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
    //    Ort::KeyValuePairs ep_metadata;
    //    Ort::KeyValuePairs ep_options;
    //    ep_metadata.Add("supported_devices", "CrackGriffin 7+");
    //    ep_options.Add("run_really_fast", "true");
    //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
    //    ep_devices[num_ep_devices++] = ep_device.release();
    //  }
    //}
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* /*devices*/,
                                                       const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                       size_t num_devices,
                                                       const OrtSessionOptions* session_options,
                                                       const OrtLogger* logger,
                                                       OrtEp** ep) noexcept {
  auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for CPU and only expected to be selected for one CPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
    // the EP has been selected for.
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Example EP only supports selection for one device.");
  }

  // Create the execution provider
  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  // Create EP configuration from session options, if needed.
  // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
  std::string ep_context_enable;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(factory->ort_api, *session_options,
                                                 "ep.context_enable", "0", ep_context_enable));

  ExampleEp::Config config = {};
  config.enable_ep_context = ep_context_enable == "1";

  auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, config, *logger);

  *ep = dummy_ep.release();
  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
  delete dummy_ep;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                              const OrtMemoryInfo* memory_info,
                                                              const OrtKeyValuePairs* allocator_options,
                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *allocator = nullptr;

  if (memory_info != factory.default_memory_info_.get()) {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  OrtDeviceMemoryType mem_type;
  RETURN_IF_ERROR(factory.ort_api.MemoryInfoGetDeviceMemType(memory_info, &mem_type));

  if (mem_type != OrtDeviceMemoryType_DEFAULT) {
    // we only registered default memory with EpDevice_AddAllocatorInfo so this should never happen
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected OrtDeviceMemoryType.");
  }

  // NOTE: The factory implementation is free to return a shared OrtAllocator* instance instead of creating a new
  //       allocator on each call. To do this have an allocator instance as an OrtEpFactory class member and make
  //       ReleaseAllocatorImpl a no-op.
  //
  // NOTE: EP should implement it's own arena logic. ep_arena.cc/h is provided as a reference and we use it here for
  //       device memory. `allocator_options` can be used for arena configuration and there is a helper in ep_arena.h
  //       to convert from OrtKeyValuePairs to the same arena config settings that ORT uses.
  //       You are of course free to have completely different settings.
  // create/use the shared arena based allocator
  if (!factory.arena_allocator_) {
    // initial shared allocator in environment does not have allocator options.
    // if the user calls CreateSharedAllocator they can provide options to configure the arena differently.
    factory.arena_allocator_using_default_settings_ = allocator_options == nullptr;
    auto allocator = std::make_unique<CustomAllocator>(memory_info, factory);
    RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(allocator), allocator_options, factory.ort_api,
                                                            factory.logger_, factory.arena_allocator_));

  } else {
    if (factory.arena_allocator_using_default_settings_ && allocator_options) {
      // potential change in arena settings. up to EP author to determine how to handle this.
      // we should not get here if replacing the shared allocator in the environment, as we free the existing one
      // before replacing it. i.e. ReleaseAllocatorImpl should have been called, and arena_allocator_ should be null.
    }

    // to keep this example simple we create a non-arena based allocator
    auto allocator = std::make_unique<CustomAllocator>(memory_info, factory);
    *allocator = allocator.release();
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  if (allocator == factory.arena_allocator_.get()) {
    factory.arena_allocator_ = nullptr;
  } else {
    delete static_cast<CustomAllocator*>(allocator);
  }
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();

  return nullptr;
}

/*static*/
bool ORT_API_CALL ExampleEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // the example EP implements stream synchronization.
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                        const OrtMemoryDevice* memory_device,
                                                                        const OrtKeyValuePairs* stream_options,
                                                                        OrtSyncStreamImpl** stream) noexcept {
  auto& factory = *static_cast<const ExampleEpFactory*>(this_ptr);
  *stream = nullptr;

  // we only need stream synchronization on the device stream
  if (factory.ep_api.MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_DEFAULT) {
    auto sync_stream = std::make_unique<StreamImpl>(factory, /*OrtEp**/ nullptr, stream_options);
    *stream = sync_stream.release();
  }

  return nullptr;
}
