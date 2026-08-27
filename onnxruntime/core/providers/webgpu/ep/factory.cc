// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "factory.h"
#include "ep.h"

#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <optional>
#include <vector>

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

namespace {
constexpr const char* kWebGpuDeviceIdMetadata = "webgpu_device_id";
constexpr const char* kWebGpuAdapterLuidMetadata = "webgpu_adapter_luid";
constexpr const char* kHardwareDeviceLuidMetadata = "LUID";

std::optional<uint64_t> GetHardwareDeviceLuid(const OrtHardwareDevice& device) {
  const OrtKeyValuePairs* metadata = Api().ort.HardwareDevice_Metadata(&device);
  const char* luid_value = metadata == nullptr ? nullptr : Api().ort.GetKeyValue(metadata, kHardwareDeviceLuidMetadata);
  if (luid_value == nullptr) {
    return std::nullopt;
  }

  uint64_t luid = 0;
  const char* end = luid_value + std::strlen(luid_value);
  const auto result = std::from_chars(luid_value, end, luid);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }

  return luid;
}

struct SharedAllocatorContextLease {
  SharedAllocatorContextLease(int context_id_in, std::optional<uint64_t> adapter_luid_in,
                              std::optional<uint32_t> adapter_vendor_id_in,
                              std::optional<uint32_t> adapter_device_id_in)
      : context_id{context_id_in},
        adapter_luid{adapter_luid_in},
        adapter_vendor_id{adapter_vendor_id_in},
        adapter_device_id{adapter_device_id_in} {
  }

  ~SharedAllocatorContextLease() {
    if (has_context) {
      WebGpuContextFactory::ReleaseContext(context_id);
    }
  }

  const webgpu::BufferManager& GetBufferManager() {
    std::call_once(init_flag, [this]() {
      WebGpuContextConfig config{};
      config.context_id = context_id;
      config.adapter_luid = adapter_luid;
      config.adapter_vendor_id = adapter_vendor_id;
      config.adapter_device_id = adapter_device_id;
      WebGpuContextFactory::CreateContext(config);
      has_context = true;
    });
    return WebGpuContextFactory::GetContext(context_id).BufferManager();
  }

  int context_id;
  std::optional<uint64_t> adapter_luid;
  std::optional<uint32_t> adapter_vendor_id;
  std::optional<uint32_t> adapter_device_id;
  std::once_flag init_flag;
  bool has_context{false};
};
}  // namespace

Factory::DeviceEntry::DeviceEntry(int device_id_in, std::optional<uint64_t> adapter_luid_in,
                                  uint32_t adapter_vendor_id_in, uint32_t adapter_device_id_in)
    : device_id{device_id_in},
      adapter_luid{adapter_luid_in},
      adapter_vendor_id{adapter_luid.has_value() ? std::make_optional(adapter_vendor_id_in) : std::nullopt},
      adapter_device_id{adapter_luid.has_value() ? std::make_optional(adapter_device_id_in) : std::nullopt},
      default_memory_info{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
                          0, static_cast<uint32_t>(device_id), OrtDeviceMemoryType_DEFAULT,
                          0, OrtDeviceAllocator},
      readonly_memory_info{WEBGPU_BUFFER, OrtMemoryInfoDeviceType_GPU,
                           0, static_cast<uint32_t>(device_id), OrtDeviceMemoryType_DEFAULT,
                           0, OrtReadOnlyAllocator} {
}

Factory::DeviceEntry* Factory::FindDeviceEntry(int device_id) const {
  const auto entry = std::find_if(device_entries_.begin(), device_entries_.end(),
                                  [device_id](const auto& candidate) {
                                    return candidate->device_id == device_id;
                                  });
  return entry == device_entries_.end() ? nullptr : entry->get();
}

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

  struct Candidate {
    const OrtHardwareDevice* hardware_device;
    std::optional<uint64_t> adapter_luid;
  };
  std::vector<Candidate> candidates;
  const OrtHardwareDevice* first_gpu = nullptr;
  for (size_t i = 0; i < num_devices; ++i) {
    if (Api().ort.HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      continue;
    }
    if (first_gpu == nullptr) {
      first_gpu = devices[i];
    }

#if defined(_WIN32) && defined(DAWN_ENABLE_D3D12)
    auto adapter_luid = GetHardwareDeviceLuid(*devices[i]);
    if (adapter_luid.has_value()) {
      candidates.push_back({devices[i], adapter_luid});
    }
#else
    if (candidates.empty()) {
      candidates.push_back({devices[i], std::nullopt});
    }
#endif
  }

  if (candidates.empty() && first_gpu != nullptr) {
    candidates.push_back({first_gpu, std::nullopt});
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
    return left.adapter_luid.value_or(0) < right.adapter_luid.value_or(0);
  });

  for (size_t i = 0; i < candidates.size() && num_ep_devices < max_ep_devices; ++i) {
    const int device_id = static_cast<int>(i);
    DeviceEntry* entry = factory->FindDeviceEntry(device_id);
    if (entry == nullptr) {
      const uint32_t adapter_vendor_id = Api().ort.HardwareDevice_VendorId(candidates[i].hardware_device);
      const uint32_t adapter_device_id = Api().ort.HardwareDevice_DeviceId(candidates[i].hardware_device);
      factory->device_entries_.push_back(std::make_unique<DeviceEntry>(
          device_id, candidates[i].adapter_luid, adapter_vendor_id, adapter_device_id));
      entry = factory->device_entries_.back().get();
    } else if (entry->adapter_luid != candidates[i].adapter_luid) {
      return Api().ort.CreateStatus(ORT_FAIL, "WebGPU hardware device ordering changed during enumeration.");
    }

    OrtKeyValuePairs* ep_metadata = nullptr;
    Api().ort.CreateKeyValuePairs(&ep_metadata);
    Api().ort.AddKeyValuePair(ep_metadata, kWebGpuDeviceIdMetadata, std::to_string(device_id).c_str());
    if (entry->adapter_luid.has_value()) {
      Api().ort.AddKeyValuePair(ep_metadata, kWebGpuAdapterLuidMetadata,
                                std::to_string(*entry->adapter_luid).c_str());
    }

    OrtEpDevice* ep_device = nullptr;
    OrtStatus* status = Api().ep.CreateEpDevice(this_ptr, candidates[i].hardware_device,
                                                ep_metadata, nullptr, &ep_device);
    Api().ort.ReleaseKeyValuePairs(ep_metadata);
    ORT_API_RETURN_IF_ERROR(status);
    ORT_API_RETURN_IF_ERROR(Api().ep.EpDevice_AddAllocatorInfo(ep_device, entry->default_memory_info));
    ORT_API_RETURN_IF_ERROR(Api().ep.EpDevice_AddAllocatorInfo(ep_device, entry->readonly_memory_info));
    ep_devices[num_ep_devices++] = ep_device;
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
    const OrtKeyValuePairs* const* ep_metadata,
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

  if (!selected_virtual_device) {
    if (ep_metadata == nullptr || ep_metadata[0] == nullptr) {
      return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT, "WebGPU EP requires per-device metadata.");
    }
    const char* device_id = Api().ort.GetKeyValue(ep_metadata[0], kWebGpuDeviceIdMetadata);
    if (device_id == nullptr) {
      return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT, "WebGPU EP metadata is missing the device ID.");
    }
    auto status = config_options.AddConfigEntry(options::kDeviceId, device_id);
    if (!status.IsOK()) {
      return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
    }
    if (const char* adapter_luid = Api().ort.GetKeyValue(ep_metadata[0], kWebGpuAdapterLuidMetadata);
        adapter_luid != nullptr) {
      status = config_options.AddConfigEntry(options::kAdapterLuid, adapter_luid);
      if (!status.IsOK()) {
        return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
      }
    }
    if (Api().ort.GetKeyValue(ep_metadata[0], kWebGpuAdapterLuidMetadata) != nullptr) {
      status = config_options.AddConfigEntry(
          options::kAdapterVendorId, std::to_string(Api().ort.HardwareDevice_VendorId(devices[0])).c_str());
      if (!status.IsOK()) {
        return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
      }
      status = config_options.AddConfigEntry(
          options::kAdapterDeviceId, std::to_string(Api().ort.HardwareDevice_DeviceId(devices[0])).c_str());
      if (!status.IsOK()) {
        return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
      }
    }
  }

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
      [webgpu_ep_ptr]() -> const webgpu::BufferManager& { return webgpu_ep_ptr->BufferManager(); }, false,
      [webgpu_ep_ptr]() { return !webgpu_ep_ptr->IsRunActive(); });
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
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Ort::ConstMemoryInfo ort_memory_info{memory_info};

  if (ort_memory_info.GetAllocatorType() != OrtDeviceAllocator ||
      ort_memory_info.GetAllocatorName() != WEBGPU_BUFFER) {
    return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT,
                                  "Unsupported memory info for shared allocator.");
  }

  auto* factory = static_cast<Factory*>(this_ptr);
  const int device_id = ort_memory_info.GetDeviceId();
  const DeviceEntry* entry = factory->FindDeviceEntry(device_id);
  if (entry == nullptr) {
    return Api().ort.CreateStatus(ORT_INVALID_ARGUMENT, "Unknown WebGPU device ID for shared allocator.");
  }
  const std::optional<uint64_t> adapter_luid = entry->adapter_luid;
  const auto context_lease = std::make_shared<SharedAllocatorContextLease>(
      device_id, adapter_luid, entry->adapter_vendor_id, entry->adapter_device_id);

  auto allocator_impl = std::make_shared<webgpu::GpuBufferAllocator>(
      [context_lease]() -> const webgpu::BufferManager& {
        return context_lease->GetBufferManager();
      },
      false,
      []() { return true; });
  *allocator = new onnxruntime::ep::adapter::Allocator(memory_info, std::move(allocator_impl));
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
  *data_transfer = OrtWebGpuCreateDataTransfer(/*context_id*/ -1);
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
