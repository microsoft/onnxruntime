// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "factory.h"
#include "ep.h"

#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"

#include "core/framework/execution_provider.h"
#include "core/framework/config_options.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/allocator.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

// Constructor
Factory::Factory() : OrtEpFactory{},
                     default_memory_info_{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
                                          0,  // vendor id
                                          0,  // device id
                                          OrtDeviceMemoryType_DEFAULT,
                                          0,  // alignment
                                          OrtDeviceAllocator},
                     readonly_memory_info_{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
                                           0,  // vendor id
                                           0,  // device id
                                           OrtDeviceMemoryType_DEFAULT,
                                           0,  // alignment
                                           OrtReadOnlyAllocator} {
  ort_version_supported = ORT_API_VERSION;

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
}

// Static C API implementations

const char* ORT_API_CALL Factory::GetNameImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kWebGpuExecutionProvider;
}

const char* ORT_API_CALL Factory::GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return "Microsoft";
}

uint32_t ORT_API_CALL Factory::GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return 0;
}

const char* ORT_API_CALL Factory::GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return ORT_PLUGIN_EP_VERSION;
}

OrtStatus* ORT_API_CALL Factory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto factory = static_cast<Factory*>(this_ptr);

  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (Api().ort.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // TODO: any metadata or options to add?
      OrtEpDevice* ep_device = nullptr;
      ORT_API_RETURN_IF_ERROR(Api().ep.CreateEpDevice(this_ptr,
                                                      &device, nullptr, nullptr,
                                                      &ep_device));
      ORT_API_RETURN_IF_ERROR(Api().ep.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));
      ORT_API_RETURN_IF_ERROR(Api().ep.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_));
      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Factory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  if (num_devices == 0) {
    return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT, "No hardware devices provided to create WebGPU EP.");
  }

  OrtKeyValuePairs* session_config_entries = nullptr;
  ORT_API_RETURN_IF_ERROR(Api().ort.GetSessionOptionsConfigEntries(session_options, &session_config_entries));
  Ort::KeyValuePairs session_config_entries_holder(session_config_entries);  // allow automatic release

  auto config_options = ConfigOptions{};
  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  size_t num_entries = 0;
  Api().ort.GetKeyValuePairs(session_config_entries, &keys, &values, &num_entries);
  for (size_t i = 0; i < num_entries; ++i) {
    auto status = config_options.AddConfigEntry(keys[i], values[i]);
    if (!status.IsOK()) {
      return Api().ort.CreateStatus((OrtErrorCode)status.Code(), status.ErrorMessage().c_str());
    }
  }

  auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(config_options);
  auto webgpu_ep = webgpu_ep_factory->CreateProvider(*session_options, *logger);
  static_cast<WebGpuExecutionProvider*>(webgpu_ep.get())->SetEpLogger(logger);
  auto factory = static_cast<Factory*>(this_ptr);
  const int context_id = webgpu_ep->GetDeviceId();
  Ep::Config webgpu_ep_config{
      CPUAllocator::DefaultInstance(),                                                                                              // CPU allocator
      std::make_shared<webgpu::GpuBufferAllocator>(WebGpuContextFactory::GetContext(context_id).BufferManager(), false),            // default device allocator
      std::make_shared<webgpu::GpuBufferAllocator>(WebGpuContextFactory::GetContext(context_id).InitializerBufferManager(), true),  // initializer device allocator
  };
  *ep = new Ep(std::move(webgpu_ep), *factory, *logger, webgpu_ep_config);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL Factory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<Ep*>(ep);
}

OrtStatus* ORT_API_CALL Factory::CreateAllocatorImpl(
    OrtEpFactory* /*this_ptr*/,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstMemoryInfo ort_memory_info{memory_info};

  if (ort_memory_info.GetAllocatorType() != OrtDeviceAllocator ||
      ort_memory_info.GetDeviceId() != 0 ||
      ort_memory_info.GetAllocatorName() != WEBGPU_BUFFER) {
    return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "Unsupported memory info for shared allocator.");
  }

  *allocator = new onnxruntime::ep::adapter::Allocator(memory_info,
                                                       [](const OrtMemoryInfo&) -> AllocatorPtr {
                                                         return std::make_shared<webgpu::GpuBufferAllocator>(WebGpuContextFactory::DefaultContext().BufferManager(), false);
                                                       });
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL Factory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  onnxruntime::ep::adapter::Allocator* ptr = static_cast<onnxruntime::ep::adapter::Allocator*>(allocator);
  delete ptr;
}

OrtStatus* ORT_API_CALL Factory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/,
    OrtDataTransferImpl** data_transfer) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  *data_transfer = OrtWebGpuCreateDataTransfer();  // TODO(fs-eire): pass context id if needed
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

bool ORT_API_CALL Factory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;  // Default: not stream aware
}

OrtStatus* ORT_API_CALL Factory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* /*this_ptr*/,
    const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  *stream = nullptr;
  return Api().ort.CreateStatus(ORT_NOT_IMPLEMENTED,
                                "CreateSyncStreamForDevice is not implemented for this EP factory.");
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
