// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep_factory.h"
#include "cuda_ep.h"
#include "cuda_plugin_kernels.h"
#include "core/common/string_utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace onnxruntime {
namespace cuda_plugin {

CudaEpFactory::CudaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                             const OrtLogger& default_logger)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      default_logger_(default_logger) {
  ort_version_supported = ORT_API_VERSION;

  if (!::onnxruntime::ep::adapter::LoggingManager::HasDefaultLogger()) {
    ::onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(&default_logger);
  }

  // Assign callback function pointers
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

CudaEpFactory::~CudaEpFactory() {
  if (kernel_registry_ != nullptr) {
    ep_api_.ReleaseKernelRegistry(kernel_registry_);
  }
}

OrtStatus* CudaEpFactory::GetKernelRegistryForEp(CudaEp& ep,
                                                 const OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  std::lock_guard<std::mutex> lock(registry_mutex_);

  if (kernel_registry_ == nullptr) {
    const char* ep_name = ep.GetEpName();
    // CreateCudaKernelRegistry dispatches between legacy/generated registrations
    // and adapter-mode registration path based on build configuration.
    RETURN_IF_ERROR(CreateCudaKernelRegistry(ep_api_, ep_name, nullptr, &kernel_registry_));
  }

  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

// ---------------------------------------------------------------------------
// OrtEpFactory callback implementations
// ---------------------------------------------------------------------------

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL CudaEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->vendor_id_;
}

/*static*/
const char* ORT_API_CALL CudaEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  return static_cast<const CudaEpFactory*>(this_ptr)->ep_version_.c_str();
}

namespace {

std::string ToUpper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

std::string GetProviderOptionPrefix(std::string_view provider_name) {
  return "ep." + onnxruntime::utils::GetLowercaseString(std::string{provider_name}) + ".";
}

void LogWarning(const OrtApi& ort_api, const OrtLogger& logger, const char* file, int line,
                const char* function, const char* msg) {
  OrtStatus* st = ort_api.Logger_LogMessage(&logger, ORT_LOGGING_LEVEL_WARNING, msg, file, line, function);
  if (st != nullptr) {
    ort_api.ReleaseStatus(st);
  }
}

}  // namespace

CudaEpFactory::HardwareDeviceKey CudaEpFactory::MakeDeviceKey(const OrtApi& ort_api,
                                                              const OrtHardwareDevice& device) {
  return {
      ort_api.HardwareDevice_Type(&device),
      ort_api.HardwareDevice_VendorId(&device),
      ort_api.HardwareDevice_DeviceId(&device),
  };
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* hw_devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;
  auto release_ep_devices = [&](OrtStatus* status) -> OrtStatus* {
    for (size_t j = 0; j < num_ep_devices; ++j) {
      factory->ep_api_.ReleaseEpDevice(ep_devices[j]);
      ep_devices[j] = nullptr;
    }
    num_ep_devices = 0;
    return status;
  };

  int cuda_device_index = 0;
  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *hw_devices[i];
    auto hw_type = factory->ort_api_.HardwareDevice_Type(&device);

    if (hw_type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // Filter by vendor ID to avoid claiming non-NVIDIA GPUs on mixed-vendor hosts.
      // vendor_id == 0 means the hardware enumeration did not provide a vendor ID,
      // in which case we fall through and let the CUDA runtime validate the device.
      uint32_t hw_vendor_id = factory->ort_api_.HardwareDevice_VendorId(&device);
      if (hw_vendor_id != 0 && hw_vendor_id != factory->vendor_id_) {
        continue;  // Skip non-NVIDIA GPUs
      }

      // CUDA uses contiguous ordinals for CUDA-visible NVIDIA devices. Build that
      // mapping from the filtered hardware-device list instead of relying on the
      // ORT hardware device id, which is not guaranteed to be a CUDA ordinal.
      int current_device_id = cuda_device_index++;
      const auto device_key = CudaEpFactory::MakeDeviceKey(factory->ort_api_, device);
      DeviceCacheEntry* cache_entry = nullptr;
      {
        std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
        auto [it, inserted] = factory->device_cache_.try_emplace(device_key);
        if (inserted) {
          it->second.cuda_device_id = current_device_id;
          it->second.device_memory_info = Ort::MemoryInfo{"Cuda",
                                                          OrtMemoryInfoDeviceType_GPU,
                                                          factory->vendor_id_,
                                                          static_cast<uint32_t>(current_device_id),
                                                          OrtDeviceMemoryType_DEFAULT,
                                                          /*alignment is default*/ 0,
                                                          OrtAllocatorType::OrtDeviceAllocator};
          it->second.pinned_memory_info = Ort::MemoryInfo{"CudaPinned",
                                                          OrtAllocatorType::OrtDeviceAllocator,
                                                          current_device_id,
                                                          OrtMemType::OrtMemTypeCPU};
        }

        cache_entry = &it->second;
        current_device_id = cache_entry->cuda_device_id;
      }

      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);
      factory->ort_api_.AddKeyValuePair(ep_metadata, "cuda_device_id", std::to_string(current_device_id).c_str());
      factory->ort_api_.AddKeyValuePair(ep_options, "device_id", std::to_string(current_device_id).c_str());

      // Get CUDA device properties for metadata
      int cuda_device_count = 0;
      cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
      if (err == cudaSuccess && cuda_device_count > 0 && current_device_id < cuda_device_count) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, current_device_id) == cudaSuccess) {
          factory->ort_api_.AddKeyValuePair(ep_metadata, "cuda_device_name", prop.name);
          factory->ort_api_.AddKeyValuePair(
              ep_metadata, "cuda_compute_capability",
              (std::to_string(prop.major) + "." + std::to_string(prop.minor)).c_str());
        }
      }

      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ep_api_.CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                     &ep_device);
      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return release_ep_devices(status);
      }

      auto release_current_ep_device = [factory](OrtEpDevice* device) {
        factory->ep_api_.ReleaseEpDevice(device);
      };
      // ep_device_guard owns the current device. On error, release_ep_devices cleans up
      // previously committed devices [0, num_ep_devices), while the guard cleans up this one.
      std::unique_ptr<OrtEpDevice, decltype(release_current_ep_device)> ep_device_guard(ep_device, release_current_ep_device);

      // Register allocator info for GPU device memory
      status = factory->ep_api_.EpDevice_AddAllocatorInfo(ep_device, cache_entry->device_memory_info);
      if (status != nullptr) {
        return release_ep_devices(status);
      }

      // Register allocator info for pinned host memory associated with the
      // same CUDA ordinal as the device allocator above.
      status = factory->ep_api_.EpDevice_AddAllocatorInfo(ep_device, cache_entry->pinned_memory_info);
      if (status != nullptr) {
        return release_ep_devices(status);
      }

      ep_devices[num_ep_devices++] = ep_device_guard.release();
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA EP factory currently supports exactly one device per EP instance. "
        "Pass a single OrtHardwareDevice when creating the CUDA plugin EP.");
  }
  if (devices == nullptr || devices[0] == nullptr) {
    return factory->ort_api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CUDA EP factory requires a valid device.");
  }

  // Parse configuration from session options.
  // The read helpers intentionally swallow errors: if a config entry is
  // absent or malformed the default value in Config is kept.
  CudaEp::Config config{};

  {
    std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
    auto it = factory->device_cache_.find(CudaEpFactory::MakeDeviceKey(factory->ort_api_, *devices[0]));
    if (it == factory->device_cache_.end()) {
      return factory->ort_api_.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "CUDA EP factory could not resolve the requested device. "
          "Enumerate EP devices again and retry session creation.");
    }
    config.device_id = it->second.cuda_device_id;
  }

  auto try_get_session_config = [&](std::string_view key) -> std::optional<std::string> {
    if (session_options == nullptr) {
      return std::nullopt;
    }

    size_t size = 0;
    OrtStatus* status = factory->ort_api_.GetSessionConfigEntry(session_options, key.data(), nullptr, &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return std::nullopt;
    }
    if (size == 0) {
      return std::nullopt;
    }
    std::vector<char> buf(size);
    status = factory->ort_api_.GetSessionConfigEntry(session_options, key.data(), buf.data(), &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return std::nullopt;
    }
    return std::string(buf.data());
  };

  auto log_invalid_session_config = [&](std::string_view key, std::string_view expected) {
    if (logger == nullptr) {
      return;
    }

    const std::string msg = std::string("Failed to parse session config for key '") +
                            std::string(key) + "'. Expected " + std::string(expected) +
                            ". Using default value.";

    OrtStatus* st = factory->ort_api_.Logger_LogMessage(
        logger, ORT_LOGGING_LEVEL_WARNING, msg.c_str(), "cuda_ep_factory.cc", __LINE__, "CudaEpFactory");
    if (st != nullptr) {
      factory->ort_api_.ReleaseStatus(st);
    }
  };

  auto read_session_config_bool = [&](std::initializer_list<std::string_view> keys, bool& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      const auto normalized = ToUpper(*raw_value);
      if (normalized == "1" || normalized == "TRUE") {
        value = true;
        return;
      }
      if (normalized == "0" || normalized == "FALSE") {
        value = false;
        return;
      }

      log_invalid_session_config(key, "a boolean");
      return;
    }
  };

  auto read_cudnn_conv_algo = [&](std::initializer_list<std::string_view> keys, int& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      try {
        value = std::stoi(*raw_value);
        return;
      } catch (const std::exception&) {
      }

      const auto normalized = ToUpper(*raw_value);
      if (normalized == "EXHAUSTIVE") {
        value = 0;
        return;
      }
      if (normalized == "HEURISTIC") {
        value = 1;
        return;
      }
      if (normalized == "DEFAULT") {
        value = 2;
        return;
      }

      log_invalid_session_config(key, "an integer or one of EXHAUSTIVE/HEURISTIC/DEFAULT");
      return;
    }
  };

  auto read_session_config_non_negative_int = [&](std::initializer_list<std::string_view> keys, int& value) {
    for (const auto& key : keys) {
      auto raw_value = try_get_session_config(key);
      if (!raw_value.has_value()) {
        continue;
      }

      try {
        int parsed = std::stoi(*raw_value);
        if (parsed < 0) {
          log_invalid_session_config(key, "a non-negative integer");
          return;
        }

        value = parsed;
        return;
      } catch (const std::exception&) {
      }

      log_invalid_session_config(key, "a non-negative integer");
      return;
    }
  };

  const std::string ep_options_prefix = GetProviderOptionPrefix(factory->GetEpName());
  const std::string prefer_nhwc_key = ep_options_prefix + "prefer_nhwc";
  const std::string prefer_nhwc_layout_key = ep_options_prefix + "prefer_nhwc_layout";
  const std::string use_tf32_key = ep_options_prefix + "use_tf32";
  const std::string skip_layer_norm_key = ep_options_prefix + "enable_skip_layer_norm_strict_mode";
  const std::string cudnn_use_max_workspace_key = ep_options_prefix + "cudnn_conv_use_max_workspace";
  const std::string cudnn_conv1d_pad_key = ep_options_prefix + "cudnn_conv1d_pad_to_nc1d";
  const std::string cudnn_conv_algo_key = ep_options_prefix + "cudnn_conv_algo";
  const std::string cudnn_conv_algo_search_key = ep_options_prefix + "cudnn_conv_algo_search";
  const std::string fuse_conv_bias_key = ep_options_prefix + "fuse_conv_bias";
  const std::string sdpa_kernel_key = ep_options_prefix + "sdpa_kernel";

  // Prefer plugin-provider-option keys, then fall back to the legacy ep.cuda.*
  // aliases and finally to the historical flat session config names.
  read_session_config_bool(
      {prefer_nhwc_key, prefer_nhwc_layout_key, "ep.cuda.prefer_nhwc_layout", "prefer_nhwc", "prefer_nhwc_layout"},
      config.prefer_nhwc);
  read_session_config_bool({use_tf32_key, "ep.cuda.use_tf32", "use_tf32"}, config.use_tf32);
  read_session_config_bool(
      {skip_layer_norm_key, "ep.cuda.enable_skip_layer_norm_strict_mode", "enable_skip_layer_norm_strict_mode"},
      config.enable_skip_layer_norm_strict_mode);
  read_session_config_bool(
      {cudnn_use_max_workspace_key, "ep.cuda.cudnn_conv_use_max_workspace", "cudnn_conv_use_max_workspace"},
      config.cudnn_conv_use_max_workspace);
  read_session_config_bool(
      {cudnn_conv1d_pad_key, "ep.cuda.cudnn_conv1d_pad_to_nc1d", "cudnn_conv1d_pad_to_nc1d"},
      config.cudnn_conv1d_pad_to_nc1d);
  read_cudnn_conv_algo(
      {cudnn_conv_algo_search_key, cudnn_conv_algo_key, "ep.cuda.cudnn_conv_algo_search", "ep.cuda.cudnn_conv_algo",
       "cudnn_conv_algo_search", "cudnn_conv_algo"},
      config.cudnn_conv_algo);
  read_session_config_bool(
      {fuse_conv_bias_key, "ep.cuda.fuse_conv_bias", "fuse_conv_bias"},
      config.fuse_conv_bias);
  read_session_config_non_negative_int(
      {sdpa_kernel_key, "ep.cuda.sdpa_kernel", "sdpa_kernel"},
      config.sdpa_kernel);

  const OrtLogger& ep_logger = logger ? *logger : factory->default_logger_;
  auto actual_ep = std::make_unique<CudaEp>(*factory, config, ep_logger);
  *ep = actual_ep.release();

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
void ORT_API_CALL CudaEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<CudaEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  *allocator = nullptr;

  const char* name = "";
  OrtStatus* status = factory.ort_api_.MemoryInfoGetName(memory_info, &name);
  if (status != nullptr) {
    return status;
  }
  int req_device_id = 0;
  status = factory.ort_api_.MemoryInfoGetId(memory_info, &req_device_id);
  if (status != nullptr) {
    return status;
  }

  if (name != nullptr && strcmp(name, "Cuda") == 0) {
    auto cuda_allocator = std::make_unique<CudaDeviceAllocator>(memory_info, req_device_id);
    *allocator = cuda_allocator.release();
    return nullptr;
  }

  if (name != nullptr && strcmp(name, "CudaPinned") == 0) {
    auto pinned_allocator = std::make_unique<CudaPinnedAllocator>(memory_info);
    *allocator = pinned_allocator.release();
    return nullptr;
  }

  return factory.ort_api_.CreateStatus(
      ORT_INVALID_ARGUMENT,
      "Unknown memory info provided to CUDA EP CreateAllocator.");
}

/*static*/
void ORT_API_CALL CudaEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  if (!allocator) return;
  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  auto* typed_allocator = static_cast<CudaAllocatorBase*>(allocator);
  switch (typed_allocator->GetKind()) {
    case CudaAllocatorKind::kDevice:
      delete static_cast<CudaDeviceAllocator*>(allocator);
      return;
    case CudaAllocatorKind::kPinned:
      delete static_cast<CudaPinnedAllocator*>(allocator);
      return;
    default:
      LogWarning(factory->ort_api_, factory->default_logger_, __FILE__, __LINE__,
                 "CudaEpFactory::ReleaseAllocatorImpl",
                 "ReleaseAllocatorImpl received an unknown CudaAllocatorKind. Leaking the allocator instance.");
      assert(false && "Unknown CudaAllocatorKind");
      return;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  auto data_transfer_impl = std::make_unique<CudaDataTransfer>(factory.ort_api_, factory.ep_api_);
  *data_transfer = data_transfer_impl.release();
  return nullptr;
}

/*static*/
bool ORT_API_CALL CudaEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;  // CUDA EP is stream-aware
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  int req_device_id = factory->ep_api_.MemoryDevice_GetDeviceId(memory_device);
  auto cuda_stream = std::make_unique<CudaSyncStream>(*factory, req_device_id, nullptr);

  // Initialize CUDA handles (stream, cuBLAS, cuDNN)
  RETURN_IF_ERROR(cuda_stream->InitHandles());

  *stream = cuda_stream.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
