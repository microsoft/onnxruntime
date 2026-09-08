// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "factory.h"
#include "ep.h"

#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"

#include <algorithm>

#include "core/framework/execution_provider.h"
#include "core/framework/config_options.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

// Constructor
Factory::Factory(Config config)
    : OrtEpFactory{},
      config_{config},
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

  if (config_.allow_virtual_devices) {
    Ort::KeyValuePairs hw_metadata;
    hw_metadata.Add(kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
    OrtStatus* status = Api().ep.CreateHardwareDevice(OrtHardwareDeviceType::OrtHardwareDeviceType_GPU,
                                                      /*vendor_id=*/0, /*device_id=*/0,
                                                      GetVendorImpl(this), hw_metadata,
                                                      &virtual_hw_device_);
    Ort::ThrowOnError(status);
  }
}

// Destructor: release the virtual hardware device if one was created.
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

  // Fall back to advertising WebGPU against the CPU device when no GPU-backed WebGPU EP device was added.
  if (factory->config_.allow_software_adapter &&
      num_ep_devices == 0 && num_devices > 0 && max_ep_devices > 0) {
    const auto* devices_end = devices + num_devices;
    const auto* cpu_device_it = std::find_if(devices, devices_end, [](const OrtHardwareDevice* device) {
      return Api().ort.HardwareDevice_Type(device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU;
    });
    if (cpu_device_it != devices_end) {
      OrtEpDevice* ep_device = nullptr;
      ORT_API_RETURN_IF_ERROR(Api().ep.CreateEpDevice(this_ptr, *cpu_device_it, nullptr, nullptr, &ep_device));
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
  if (factory->config_.allow_virtual_devices && num_ep_devices < max_ep_devices) {
    OrtEpDevice* ep_device = nullptr;
    ORT_API_RETURN_IF_ERROR(Api().ep.CreateEpDevice(this_ptr, factory->virtual_hw_device_,
                                                    nullptr, nullptr, &ep_device));
    // No allocator info: a virtual device only backs a device-free compile-only session, which stops
    // before session-state finalization and never allocates. Leaving the memory info unset also avoids
    // ORT trying to create a shared WebGPU allocator (environment.cc) with no underlying device.
    ep_devices[num_ep_devices++] = ep_device;
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
  auto* factory = static_cast<Factory*>(this_ptr);
  const auto thread_id = std::this_thread::get_id();
  {
    std::lock_guard<std::mutex> lock{factory->creation_mutex_};
    ORT_ENFORCE(factory->env_transfer_created_, "Create the WebGPU environment data transfer before creating Sessions.");
    ORT_ENFORCE(!factory->pending_eps_.contains(thread_id),
                "A WebGPU EP on this thread is still awaiting its Session data transfer.");
  }
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
  const int context_id = webgpu_ep->GetDeviceId();
  auto* webgpu_ep_ptr = static_cast<WebGpuExecutionProvider*>(webgpu_ep.get());
  // A device-free context (compile-only session) gets a no-op allocator: a real GpuBufferAllocator
  // needs a device, and such a session stops before finalization and never allocates.
  const bool device_free = !WebGpuContextFactory::GetContext(context_id).HasDevice();
  // External Session allocations must not overlap Run. Submit clears outside Run so subsequent
  // Env copies see initialized buffers; defer clears during Run to preserve command batching.
  auto device_alloc = webgpu::CreateWebGpuAllocator(
      device_free,
      [webgpu_ep_ptr]() -> const webgpu::BufferManager& { return webgpu_ep_ptr->BufferManager(); },
      [webgpu_ep_ptr]() -> webgpu::CommandRecordingState& { return webgpu_ep_ptr->Recording(); },
      false,
      /*should_submit_zero_initialize=*/[webgpu_ep_ptr]() { return !webgpu_ep_ptr->IsRunActive(); });
  Ep::Config webgpu_ep_config{
      CPUAllocator::DefaultInstance(),  // CPU allocator
      device_alloc,                     // default device allocator
      webgpu::CreateWebGpuAllocator(
          device_free,
          [webgpu_ep_ptr]() -> const webgpu::BufferManager& {
            return webgpu_ep_ptr->InitializerBufferManager();
          },
          [webgpu_ep_ptr]() -> webgpu::CommandRecordingState& { return webgpu_ep_ptr->Recording(); },
          true),  // initializer device allocator
  };
  auto created_ep = std::make_unique<Ep>(std::move(webgpu_ep), *factory, *logger, webgpu_ep_config);
  {
    std::lock_guard<std::mutex> lock{factory->creation_mutex_};
    ORT_ENFORCE(factory->pending_eps_.try_emplace(thread_id, created_ep.get()).second,
                "A WebGPU EP on this thread is still awaiting its Session data transfer.");
  }
  *ep = created_ep.release();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL Factory::ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept {
  auto* factory = static_cast<Factory*>(this_ptr);
  {
    std::lock_guard<std::mutex> lock{factory->creation_mutex_};
    auto pending_ep = std::find_if(factory->pending_eps_.begin(), factory->pending_eps_.end(),
                                   [ep](const auto& entry) { return entry.second == ep; });
    if (pending_ep != factory->pending_eps_.end()) {
      factory->pending_eps_.erase(pending_ep);
    }
  }
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

  // Env allocations can run alongside Session execution. Direct buffer allocation avoids
  // accessing Session command recording or cached-buffer clear state.
  *allocator = new onnxruntime::ep::adapter::Allocator(
      memory_info,
      [](const OrtMemoryInfo&) -> AllocatorPtr {
        auto context = std::shared_ptr<WebGpuContext>(
            &WebGpuContextFactory::DefaultContext(),
            [](WebGpuContext*) { WebGpuContextFactory::ReleaseContext(0); });
        return std::make_shared<webgpu::ExternalGpuBufferAllocator>(std::move(context));
      });
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

void ORT_API_CALL Factory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  onnxruntime::ep::adapter::Allocator* ptr = static_cast<onnxruntime::ep::adapter::Allocator*>(allocator);
  delete ptr;
}

OrtStatus* ORT_API_CALL Factory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* factory = static_cast<Factory*>(this_ptr);
  std::lock_guard<std::mutex> lock{factory->creation_mutex_};
  // ORT currently creates the Env transfer first for each factory. This ordering is not an
  // EP API guarantee; the Env transfer uses local encoders and is not bound to a Session.
  if (!factory->env_transfer_created_) {
    *data_transfer = OrtWebGpuCreateDataTransfer();
    factory->env_transfer_created_ = true;
  } else {
    // Session creation calls both callbacks on the same thread. Bind once to the owning EP;
    // subsequent copies use its recording regardless of which thread runs the Session.
    auto pending_ep = factory->pending_eps_.find(std::this_thread::get_id());
    ORT_ENFORCE(pending_ep != factory->pending_eps_.end(),
                "Create the WebGPU Session data transfer on the same thread that created its EP.");
    auto* ep = static_cast<WebGpuExecutionProvider*>(pending_ep->second->EpImpl());
    *data_transfer = OrtWebGpuCreateDataTransfer(ep->GetDeviceId(), ep);
    factory->pending_eps_.erase(pending_ep);
  }
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
