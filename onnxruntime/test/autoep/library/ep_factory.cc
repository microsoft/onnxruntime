// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory.h"

#include <cassert>

#include "ep.h"
#include "ep_allocator.h"
#include "ep_data_transfer.h"

ExampleEpFactory::ExampleEpFactory(const char* ep_name, ApiPtrs apis)
    : ApiPtrs(apis), ep_name_{ep_name} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;

  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;

  CreateDataTransfer = CreateDataTransferImpl;

  // for the sake of this example we specify a CPU allocator with no arena and 1K alignment (arbitrary)
  // as well as GPU and GPU shared memory. the actual EP implementation would typically define two at most for a
  // device (one for device memory and one for shared memory for data transfer between device and CPU)

  // setup the OrtMemoryInfo instances required by the EP.
  OrtMemoryInfo* mem_info = nullptr;
  auto* status = ort_api.CreateMemoryInfo_V2("ExampleEP CPU", OrtMemoryInfoDeviceType_CPU,
                                             /*vendor*/ 0xBE57, /* device_id */ 0,
                                             OrtDeviceMemoryType_DEFAULT,
                                             /*alignment*/ 1024,
                                             OrtAllocatorType::OrtDeviceAllocator,  // no arena
                                             &mem_info);
  assert(status == nullptr);  // should never fail.

  cpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

  //
  // GPU allocator OrtMemoryInfo for example purposes
  mem_info = nullptr;
  status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU", OrtMemoryInfoDeviceType_GPU,
                                       /*vendor*/ 0xBE57, /* device_id */ 0,
                                       OrtDeviceMemoryType_DEFAULT,
                                       /*alignment*/ 0,
                                       OrtAllocatorType::OrtDeviceAllocator,
                                       &mem_info);
  assert(status == nullptr);  // should never fail.
  default_gpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

  mem_info = nullptr;
  status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU pinned", OrtMemoryInfoDeviceType_CPU,
                                       /*vendor*/ 0xBE57, /* device_id */ 0,
                                       OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                       /*alignment*/ 0,
                                       OrtAllocatorType::OrtDeviceAllocator,
                                       &mem_info);
  assert(status == nullptr);  // should never fail.
  host_accessible_gpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

  // if we were to use GPU we'd create it like this
  data_transfer_impl_ = std::make_unique<ExampleDataTransfer>(
      apis,
      ep_api.OrtMemoryInfo_GetMemoryDevice(default_gpu_memory_info_.get()),         // device memory
      ep_api.OrtMemoryInfo_GetMemoryDevice(host_accessible_gpu_memory_info_.get())  // shared memory
  );

  data_transfer_impl_.reset();  // but we're CPU only so we return nullptr for the IDataTransfer.
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
      factory->ort_api.AddKeyValuePair(ep_metadata, "version", "0.1");
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
      // in this example we register CPU info which is unnecessary unless you need to override the default ORT allocator
      // for a non-CPU EP this would be device info (GPU/NPU) and possible host accessible info.
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->cpu_memory_info_.get()));

      ep_devices[num_ep_devices++] = ep_device;
    }

    // C++ API equivalent. Throws on error.
    //{
    //  Ort::ConstHardwareDevice device(devices[i]);
    //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
    //    Ort::KeyValuePairs ep_metadata;
    //    Ort::KeyValuePairs ep_options;
    //    ep_metadata.Add("version", "0.1");
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

  auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, *session_options, *logger);

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
                                                              const OrtKeyValuePairs* /*allocator_options*/,
                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *allocator = nullptr;

  // NOTE: The factory implementation can return a shared OrtAllocator* instead of creating a new instance on each call.
  //       To do this just make ReleaseAllocatorImpl a no-op.

  // NOTE: If OrtMemoryInfo has allocator type (call MemoryInfoGetType) of OrtArenaAllocator, an ORT BFCArena
  //       will be added to wrap the returned OrtAllocator. The EP is free to implement its own arena, and if it
  //       wants to do this the OrtMemoryInfo MUST be created with an allocator type of OrtDeviceAllocator.

  // NOTE: The OrtMemoryInfo pointer should only ever be coming straight from an OrtEpDevice, and pointer based
  // matching should work.
  if (memory_info == factory.cpu_memory_info_.get()) {
    // create a CPU allocator. use the basic OrtAllocator for this example.
    auto cpu_allocator = std::make_unique<CustomAllocator>(memory_info);
    *allocator = cpu_allocator.release();
  } else if (memory_info == factory.default_gpu_memory_info_.get()) {
    // create a GPU allocator
    return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "Example is not implemented.");
  } else if (memory_info == factory.host_accessible_gpu_memory_info_.get()) {
    // create a pinned memory allocator
    return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "Example is not implemented.");
  } else {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
  delete static_cast<CustomAllocator*>(allocator);
}

/*static*/
OrtStatus* ORT_API_CALL ExampleEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();

  return nullptr;
}
