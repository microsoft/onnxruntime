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
Factory::Factory() : OrtEpFactory{} {
  ort_version_supported = ORT_API_VERSION;

  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;

  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;

  CreateAllocator = CreateAllocatorImpl;        // TODO
  ReleaseAllocator = ReleaseAllocatorImpl;      // TODO
  CreateDataTransfer = CreateDataTransferImpl;  // TODO
}

// Static C API implementations

const char* ORT_API_CALL Factory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return kWebGpuExecutionProvider;
}

const char* ORT_API_CALL Factory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return "Microsoft";
}

uint32_t ORT_API_CALL Factory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  return 0;
}

const char* ORT_API_CALL Factory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return "0.1.0";
}

OrtStatus* ORT_API_CALL Factory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (Api().ort.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // TODO: any metadata or options to add?
      ORT_API_RETURN_IF_ERROR(Api().ep.CreateEpDevice(this_ptr,
                                                      &device, nullptr, nullptr,
                                                      &ep_devices[num_ep_devices++]));
    }
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL Factory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
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

  try {
    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(config_options);
    auto webgpu_ep = webgpu_ep_factory->CreateProvider(*session_options, *logger);
    auto webgpu_ep_impl = static_cast<WebGpuExecutionProvider*>(webgpu_ep.release());
    int device_id = webgpu_ep_impl->GetDeviceId();
    auto& webgpu_context = WebGpuContextFactory::GetContext(device_id);
    *ep = new Ep(webgpu_ep_impl,
                 *(static_cast<Factory*>(this_ptr)),
                 *logger,
                 {
                     CPUAllocator::DefaultInstance(),                                                                // CPU allocator
                     std::make_shared<webgpu::GpuBufferAllocator>(webgpu_context.BufferManager(), false),            // default device allocator
                     std::make_shared<webgpu::GpuBufferAllocator>(webgpu_context.InitializerBufferManager(), true),  // initializer device allocator
                 });
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }
}

void ORT_API_CALL Factory::ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept {
  delete static_cast<Ep*>(ep);
}

OrtStatus* ORT_API_CALL Factory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options,
    OrtAllocator** allocator) noexcept {
  *allocator = nullptr;
  return Api().ort.CreateStatus(ORT_NOT_IMPLEMENTED,
                                "CreateAllocator is not implemented for this EP factory.");
}

void ORT_API_CALL Factory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  onnxruntime::ep::Allocator* ptr = static_cast<onnxruntime::ep::Allocator*>(allocator);
  delete ptr;
}

OrtStatus* ORT_API_CALL Factory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  try {
    *data_transfer = OrtWebGpuCreateDataTransfer();  // TODO(fs-eire): pass context id if needed
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }
}

bool ORT_API_CALL Factory::IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept {
  return false;  // Default: not stream aware
}

OrtStatus* ORT_API_CALL Factory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* stream_options,
    OrtSyncStreamImpl** stream) noexcept {
  *stream = nullptr;
  return Api().ort.CreateStatus(ORT_NOT_IMPLEMENTED,
                                "CreateSyncStreamForDevice is not implemented for this EP factory.");
}

OrtStatus* ORT_API_CALL Factory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept {
  *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  return nullptr;
}

OrtStatus* ORT_API_CALL Factory::SetEnvironmentOptionsImpl(
    OrtEpFactory* this_ptr,
    const OrtKeyValuePairs* options) noexcept {
  return nullptr;  // Default implementation does nothing
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
