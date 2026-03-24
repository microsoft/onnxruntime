// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep_factory.h"
#include "cuda_ep.h"
#include "cuda_plugin_kernels.h"

namespace onnxruntime {
namespace cuda_plugin {

CudaEpFactory::CudaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                             const OrtLogger& default_logger)
    : OrtEpFactory{},
      ort_api_(ort_api),
      ep_api_(ep_api),
      default_logger_(default_logger),
      default_memory_info_{nullptr},
      pinned_memory_info_{nullptr} {
  ort_version_supported = ORT_API_VERSION;

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

  // Initialize default memory info for CUDA device memory.
  // The NVIDIA PCI vendor ID (0x10DE) is used to identify the device type.
  default_memory_info_ = Ort::MemoryInfo{"Cuda",
                                         OrtMemoryInfoDeviceType_GPU,
                                         vendor_id_,
                                         static_cast<uint32_t>(device_id_),
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator};

  // Initialize pinned (host accessible) memory info
  pinned_memory_info_ = Ort::MemoryInfo{"CudaPinned",
                                        OrtAllocatorType::OrtDeviceAllocator,
                                        0,
                                        OrtMemType::OrtMemTypeCPU};
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

      int32_t current_device_id = factory->ort_api_.HardwareDevice_DeviceId(&device);

      OrtKeyValuePairs* ep_metadata = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);

      // Get CUDA device properties for metadata
      int cuda_device_count = 0;
      cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
      if (err == cudaSuccess && cuda_device_count > 0 && current_device_id < cuda_device_count) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, current_device_id);
        factory->ort_api_.AddKeyValuePair(ep_metadata, "cuda_device_name", prop.name);
        factory->ort_api_.AddKeyValuePair(
            ep_metadata, "cuda_compute_capability",
            (std::to_string(prop.major) + "." + std::to_string(prop.minor)).c_str());
      }

      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ep_api_.CreateEpDevice(factory, &device, ep_metadata, nullptr,
                                                     &ep_device);
      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);

      if (status != nullptr) {
        return status;
      }

      Ort::MemoryInfo device_memory_info{"Cuda",
                                         OrtMemoryInfoDeviceType_GPU,
                                         factory->vendor_id_,
                                         static_cast<uint32_t>(current_device_id),
                                         OrtDeviceMemoryType_DEFAULT,
                                         /*alignment is default*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator};

      // Register allocator info for GPU device memory
      RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
          ep_device, device_memory_info));

      // Register allocator info for CPU pinned memory (host accessible)
      RETURN_IF_ERROR(factory->ep_api_.EpDevice_AddAllocatorInfo(
          ep_device, factory->pinned_memory_info_));

      ep_devices[num_ep_devices++] = ep_device;
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
        "CUDA EP factory currently supports only one device at a time.");
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
  config.device_id = factory->ort_api_.HardwareDevice_DeviceId(devices[0]);

  auto read_session_config_bool = [&](const char* key, bool& value) {
    size_t size = 0;
    OrtStatus* status = factory->ort_api_.GetSessionConfigEntry(session_options, key, nullptr, &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return;
    }
    if (size == 0) return;
    std::vector<char> buf(size);
    status = factory->ort_api_.GetSessionConfigEntry(session_options, key, buf.data(), &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return;
    }
    const std::string val(buf.data());
    value = (val == "1" || val == "true");
  };

  auto read_session_config_int = [&](const char* key, int& value) {
    size_t size = 0;
    OrtStatus* status = factory->ort_api_.GetSessionConfigEntry(session_options, key, nullptr, &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return;
    }
    if (size == 0) return;
    std::vector<char> buf(size);
    status = factory->ort_api_.GetSessionConfigEntry(session_options, key, buf.data(), &size);
    if (status != nullptr) {
      Ort::Status s(status);
      return;
    }
    try {
      value = std::stoi(buf.data());
    } catch (...) {
    }
  };

  // Read from flat keys first, then from ep.cuda.* prefixed keys.
  // The second pass intentionally overwrites the first so that
  // ep.cuda.* takes precedence over unprefixed keys.
  read_session_config_bool("prefer_nhwc", config.prefer_nhwc);
  read_session_config_bool("use_tf32", config.use_tf32);
  read_session_config_bool("enable_skip_layer_norm_strict_mode", config.enable_skip_layer_norm_strict_mode);
  read_session_config_bool("cudnn_conv1d_pad_to_nc1d", config.cudnn_conv1d_pad_to_nc1d);
  read_session_config_int("cudnn_conv_algo", config.cudnn_conv_algo);

  read_session_config_bool("ep.cuda.prefer_nhwc_layout", config.prefer_nhwc);
  read_session_config_bool("ep.cuda.use_tf32", config.use_tf32);
  read_session_config_bool("ep.cuda.enable_skip_layer_norm_strict_mode", config.enable_skip_layer_norm_strict_mode);
  read_session_config_bool("ep.cuda.cudnn_conv1d_pad_to_nc1d", config.cudnn_conv1d_pad_to_nc1d);
  read_session_config_int("ep.cuda.cudnn_conv_algo", config.cudnn_conv_algo);

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

  if (strcmp(name, "Cuda") == 0) {
    auto cuda_allocator = std::make_unique<CudaDeviceAllocator>(memory_info, req_device_id);
    *allocator = cuda_allocator.release();
    return nullptr;
  }

  if (strcmp(name, "CudaPinned") == 0) {
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
    OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  // We know the allocator was created by us, so cast and delete.
  // OrtAllocator itself has no Release method.
  delete allocator;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<CudaEpFactory*>(this_ptr);
  const OrtMemoryDevice* gpu_device = factory.ep_api_.MemoryInfo_GetMemoryDevice(factory.default_memory_info_);
  auto data_transfer_impl = std::make_unique<CudaDataTransfer>(factory.ort_api_, factory.ep_api_, gpu_device);
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
    const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* factory = static_cast<CudaEpFactory*>(this_ptr);
  auto cuda_stream = std::make_unique<CudaSyncStream>(*factory, factory->device_id_, nullptr);

  // Initialize CUDA handles (stream, cuBLAS, cuDNN)
  RETURN_IF_ERROR(cuda_stream->InitHandles());

  *stream = cuda_stream.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
