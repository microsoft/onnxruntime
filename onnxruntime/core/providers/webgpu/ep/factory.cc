// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "factory.h"
#include "ep.h"

#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"

#include <cstring>

#include "core/framework/execution_provider.h"
#include "core/framework/config_options.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/allocator.h"
#include "core/session/onnxruntime_env_config_keys.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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

// Destructor: release the virtual hardware device if one was created in GetSupportedDevices.
Factory::~Factory() {
  if (virtual_hw_device_ != nullptr) {
    Api().ep.ReleaseHardwareDevice(virtual_hw_device_);
    virtual_hw_device_ = nullptr;
  }
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

  // If the environment allows virtual devices, register a virtual GPU EP device (vendor/device id 0) so
  // the WebGPU EP stays selectable for a device-free compile-only session on hosts where OS device
  // enumeration finds no GPU (e.g. a Win32k-lockdown sandbox). It is offered *in addition* to any real
  // GPU device, so the device-free path remains exercisable on a host that also has a real GPU. Since
  // allow_virtual_devices is opt-in, normal (real GPU) usage is unaffected.
  if (num_ep_devices < max_ep_devices) {
    OrtKeyValuePairs* env_config = nullptr;
    ORT_API_RETURN_IF_ERROR(Api().ep.GetEnvConfigEntries(&env_config));
    Ort::KeyValuePairs env_config_holder(env_config);  // allow automatic release
    const char* allow_virtual = env_config_holder.GetValue(kOrtEnvAllowVirtualDevices);
    const bool allow_virtual_devices = allow_virtual != nullptr && std::strcmp(allow_virtual, "1") == 0;

    if (allow_virtual_devices) {
      OrtKeyValuePairs* hw_metadata = nullptr;
      Api().ort.CreateKeyValuePairs(&hw_metadata);
      Api().ort.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
      OrtStatus* status = Api().ep.CreateHardwareDevice(OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
                                                        /*vendor_id=*/0, /*device_id=*/0,
                                                        GetVendorImpl(this_ptr), hw_metadata,
                                                        &factory->virtual_hw_device_);
      Api().ort.ReleaseKeyValuePairs(hw_metadata);  // ORT makes a copy
      ORT_API_RETURN_IF_ERROR(status);

      OrtEpDevice* ep_device = nullptr;
      ORT_API_RETURN_IF_ERROR(Api().ep.CreateEpDevice(this_ptr, factory->virtual_hw_device_,
                                                      nullptr, nullptr, &ep_device));
      // No allocator info: a virtual device only backs a device-free compile-only session, which stops
      // before session-state finalization and never allocates. Leaving the memory info unset also avoids
      // ORT trying to create a shared WebGPU allocator (environment.cc) with no underlying device.
      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Factory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  if (num_devices != 1) {
    return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "WebGPU EP factory currently only supports one device at a time.");
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

  // A virtual GPU device has no real GPU behind it, so it can only back a device-free compile-only session
  // (see the concept map in webgpu_context.cc). Reject the invalid combination up front with a clear message
  // instead of letting Dawn fail obscurely when it later tries to create a device.
  const bool compile_only = config_options.GetConfigOrDefault(kOrtSessionOptionCompileOnly, "0") == "1";
  const OrtKeyValuePairs* device_metadata = Api().ort.HardwareDevice_Metadata(devices[0]);
  const bool selected_virtual_device =
      device_metadata != nullptr &&
      Api().ort.GetKeyValue(device_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual) != nullptr;
  if (selected_virtual_device && !compile_only) {
    return Api().ort.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "WebGPU EP was selected on a virtual GPU device, which has no real GPU behind it and can only serve "
        "a compile-only session (session.compile_only=1). Select a real GPU device to run inference.");
  }

  auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(config_options);
  auto webgpu_ep = webgpu_ep_factory->CreateProvider(*session_options, *logger);
  static_cast<WebGpuExecutionProvider*>(webgpu_ep.get())->SetEpLogger(logger);
  auto factory = static_cast<Factory*>(this_ptr);
  const int context_id = webgpu_ep->GetDeviceId();
  auto* webgpu_ep_ptr = static_cast<WebGpuExecutionProvider*>(webgpu_ep.get());
  // A device-free context (compile-only session) gets a no-op allocator: a real GpuBufferAllocator
  // needs a device, and such a session stops before finalization and never allocates.
  const bool device_free = !WebGpuContextFactory::GetContext(context_id).HasDevice();
  auto device_alloc = webgpu::CreateWebGpuAllocator(
      device_free,
      [webgpu_ep_ptr]() -> const webgpu::BufferManager& { return webgpu_ep_ptr->BufferManager(); }, false);
  Ep::Config webgpu_ep_config{
      CPUAllocator::DefaultInstance(),  // CPU allocator
      device_alloc,                     // default device allocator
      webgpu::CreateWebGpuAllocator(
          device_free,
          [context_id]() -> const webgpu::BufferManager& {
            return WebGpuContextFactory::GetContext(context_id).InitializerBufferManager();
          },
          true),  // initializer device allocator
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
                                                         return std::make_shared<webgpu::GpuBufferAllocator>(
                                                             []() -> const webgpu::BufferManager& {
                                                               return WebGpuContextFactory::DefaultContext()
                                                                   .BufferManager();
                                                             },
                                                             false);
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
