// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include <string.h>
#include <atomic>

#include "core/providers/shared_library/provider_api.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_custom_ops.h"
#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_utils.h"
#include "core/providers/cuda/cuda_stream_handle.h"

#include "onnx_ctx_model_helper.h"
#include "nv_provider_factory.h"
#include "nv_execution_provider.h"
#include "nv_provider_factory_creator.h"
#include "nv_data_transfer.h"
#include "nv_allocator.h"

using namespace onnxruntime;

// External declarations
namespace onnxruntime {
extern TensorrtLogger& GetTensorrtLogger(bool verbose_log);
}

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_Nv_Impl final : ProviderInfo_Nv {
  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) override {
    common::Status status = CreateTensorRTCustomOpDomainList(domain_list, extra_plugin_lib_paths);
    if (!status.IsOK()) {
      return CreateStatus(ORT_FAIL, "[NvTensorRTRTX EP] Can't create custom ops for TRT plugins.");
    }
    return nullptr;
  }

  OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) override {
    ReleaseTensorRTCustomOpDomainList(domain_list);
    return nullptr;
  }
} g_info;

struct NvProviderFactory : IExecutionProviderFactory {
  NvProviderFactory(const NvExecutionProviderInfo& info) : info_{info} {}
  ~NvProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger);

 private:
  NvExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider() {
  return std::make_unique<NvExecutionProvider>(info_);
}

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider(const OrtSessionOptions& session_options, const OrtLogger& session_logger) {
  const ConfigOptions& config_options = session_options.GetConfigOptions();
  const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

  // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
  // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
  // We extract those EP options to create a new "provider options" key/value map.
  std::string lowercase_ep_name = kNvTensorRTRTXExecutionProvider;
  std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  ProviderOptions provider_options;
  std::string key_prefix = "ep.";
  key_prefix += lowercase_ep_name;
  key_prefix += ".";

  for (const auto& [key, value] : config_options_map) {
    if (key.rfind(key_prefix, 0) == 0) {
      provider_options[key.substr(key_prefix.size())] = value;
    }
  }
  NvExecutionProviderInfo info = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(provider_options, config_options);

  auto ep = std::make_unique<NvExecutionProvider>(info);
  ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
  return ep;
}

struct Nv_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    NvExecutionProviderInfo info;
    info.device_id = device_id;

    return std::make_shared<NvProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    NvExecutionProviderInfo info = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(*provider_options, *config_options);
    return std::make_shared<NvProviderFactory>(info);
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t /*num_devices*/,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);
    auto ep_factory = CreateExecutionProviderFactory(arg);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
    // ensure that there is no longer an active context which will be destroyed when unloading cudart
    CUcontext cu_context = 0;
    CU_CALL_THROW(cuCtxGetCurrent(&cu_context));
    if (cu_context) {
      CU_CALL_THROW(cuCtxPopCurrent(&cu_context));
    }
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

//
// Plug-in EP infrastructure
//

#include "core/session/abi_devices.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

struct ErrorHelper {
  static const OrtApi* ort_api;

  static OrtStatus* ToOrtStatus(const Status& status) {
    if (status.IsOK()) {
      return nullptr;  // no error
    }

    return ort_api->CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                 status.ErrorMessage().c_str());
  }
};

const OrtApi* ErrorHelper::ort_api = nullptr;

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF_STATUS_NOTOK(fn)              \
  do {                                          \
    Status _status = (fn);                      \
    if (!_status.IsOK()) {                      \
      return ErrorHelper::ToOrtStatus(_status); \
    }                                           \
  } while (0)

#define CUDA_RETURN_IF_ERROR(expr) RETURN_IF_STATUS_NOTOK(CUDA_CALL(expr))

struct NvTrtRtxOrtAllocator : OrtAllocator {
  NvTrtRtxOrtAllocator(const OrtMemoryInfo* mem_info, const OrtApi& api) : memory_info_{mem_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    AllocOnStream = AllocOnStreamImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;  // no special behavior for Reserve so use AllocImpl
    GetStats = nullptr;   // GetStatsImpl. The CUDA allocators don't have stats currently so we can skip.

    const OrtEpApi& ep_api = *api.GetEpApi();
    const OrtMemoryDevice* mem_device = ep_api.MemoryInfo_GetMemoryDevice(mem_info);
    uint32_t device_id = ep_api.MemoryDevice_GetDeviceId(mem_device);
    const char* name = nullptr;
    auto* status = api.MemoryInfoGetName(mem_info, &name);
    static_cast<void>(status);  // GetName never fails

    if (ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
      allocator_ = std::make_unique<CUDAPinnedAllocator>(device_id, name);
    } else {
      allocator_ = std::make_unique<CUDAAllocator>(device_id, name);
    }
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    auto& impl = *static_cast<NvTrtRtxOrtAllocator*>(this_);
    return impl.allocator_->Alloc(size);
  }

  static void* ORT_API_CALL AllocOnStreamImpl(struct OrtAllocator* this_, size_t size, OrtSyncStream* stream) {
    auto& impl = *static_cast<NvTrtRtxOrtAllocator*>(this_);
    return impl.allocator_->AllocOnStream(size, stream);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    auto& impl = *static_cast<NvTrtRtxOrtAllocator*>(this_);
    impl.allocator_->Free(p);
  }

  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const NvTrtRtxOrtAllocator& impl = *static_cast<const NvTrtRtxOrtAllocator*>(this_);
    return impl.memory_info_;
  }

 private:
  const OrtMemoryInfo* memory_info_;
  std::unique_ptr<IAllocator> allocator_;
};

struct NvTrtRtxDataTransferImpl : OrtDataTransferImpl {
  NvTrtRtxDataTransferImpl(const OrtApi& ort_api_in)
      : ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                          const OrtMemoryDevice* src_memory_device,
                          const OrtMemoryDevice* dst_memory_device) noexcept {
    const auto& impl = *static_cast<const NvTrtRtxDataTransferImpl*>(this_ptr);

    // logic copied from GPUDataTransfer::CanCopy
    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
    auto src_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
    auto dst_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

    if ((src_type == OrtDevice::GPU && src_vendor_id != OrtDevice::VendorIds::NVIDIA) ||
        (dst_type == OrtDevice::GPU && dst_vendor_id != OrtDevice::VendorIds::NVIDIA)) {
      return false;
    }

    // copy must be GPU to GPU or between GPU and CPU
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
  }

  static OrtStatus* CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                    const OrtValue** src_tensors,
                                    OrtValue** dst_tensors,
                                    OrtSyncStream** streams,
                                    size_t num_tensors) noexcept {
    auto& impl = *static_cast<NvTrtRtxDataTransferImpl*>(this_ptr);
    bool need_stream_sync = false;

    for (size_t idx = 0; idx < num_tensors; ++idx) {
      const OrtValue* src_tensor = src_tensors[idx];
      OrtValue* dst_tensor = dst_tensors[idx];
      OrtSyncStream* stream = streams ? streams[idx] : nullptr;

      const OrtMemoryDevice* src_device = impl.ep_api.Value_GetMemoryDevice(src_tensor);
      const OrtMemoryDevice* dst_device = impl.ep_api.Value_GetMemoryDevice(dst_tensor);

      size_t bytes;
      RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensor, &bytes));

      const void* src_data = nullptr;
      void* dst_data = nullptr;
      RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensor, &src_data));
      RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data));

      OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
      OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);
      OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_device);
      OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_device);

      const bool src_is_gpu_default = src_type == OrtMemoryInfoDeviceType_GPU &&
                                      src_mem_type == OrtDeviceMemoryType_DEFAULT;
      const bool dst_is_gpu_default = dst_type == OrtMemoryInfoDeviceType_GPU &&
                                      dst_mem_type == OrtDeviceMemoryType_DEFAULT;

      cudaStream_t cuda_stream = nullptr;
      if (stream) {
        cuda_stream = static_cast<cudaStream_t>(impl.ort_api.SyncStream_GetHandle(stream));
      }

      if (dst_is_gpu_default) {
        if (src_is_gpu_default) {
          // Copy only if the two addresses are different.
          if (dst_data != src_data) {
            if (cuda_stream) {
              CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, cuda_stream));

            } else {
              CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));

              // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
              // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
              need_stream_sync = true;
            }
          }
        } else {
          // copy from pinned or non-pinned CPU memory to GPU
          if (cuda_stream) {
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, cuda_stream));
          } else {
            CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));

            if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
              // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not
              // have completed.
              // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
              need_stream_sync = true;
            }
          }
        }
      } else if (src_is_gpu_default) {
        // copying from GPU to CPU memory, this is blocking

        if (cuda_stream) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, cuda_stream));

        } else {
          CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
        }
      } else {
        // copying between CPU accessible memory

        if (dst_data != src_data) {
          if (cuda_stream) {
            if (src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
              // sync the stream first to make sure the data arrived
              CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
            }
          }

          memcpy(dst_data, src_data, bytes);
        }
      }
    }

    if (need_stream_sync) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
    }

    return nullptr;
  }

  static void ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
    // no-op as we have a single shared instance in OrtEpFactory which is returned from CreateDataTransferImpl, and is
    // owned by and freed by the factory.
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct NvTrtRtxSyncNotificationImpl : OrtSyncNotificationImpl {
  static OrtStatus* Create(cudaStream_t stream, const OrtApi& ort_api,
                           std::unique_ptr<NvTrtRtxSyncNotificationImpl>& notification) {
    notification.reset(new NvTrtRtxSyncNotificationImpl(stream, ort_api));  // can't use make_unique with private ctor
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&notification->event_, cudaEventDisableTiming));

    return nullptr;
  }

  static void ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    delete static_cast<NvTrtRtxSyncNotificationImpl*>(this_ptr);
  }

  static OrtStatus* ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncNotificationImpl*>(this_ptr);
    CUDA_RETURN_IF_ERROR(cudaEventRecord(impl.event_, impl.stream_));

    return nullptr;
  }

  static OrtStatus* WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                     _In_ OrtSyncStream* consumer_stream) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncNotificationImpl*>(this_ptr);

    // setup the consumer stream to wait on our event.
    void* consumer_handle = impl.ort_api.SyncStream_GetHandle(consumer_stream);
    CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_handle), impl.event_));

    return nullptr;
  }

  static OrtStatus* WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncNotificationImpl*>(this_ptr);
    CUDA_RETURN_IF_ERROR(cudaEventSynchronize(impl.event_));

    return nullptr;
  }

  ~NvTrtRtxSyncNotificationImpl() {
    cudaEventDestroy(event_);
  }

 private:
  NvTrtRtxSyncNotificationImpl(cudaStream_t stream, const OrtApi& ort_api_in)
      : stream_{stream}, ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;
    Activate = ActivateImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
    Release = ReleaseImpl;
  }

  cudaStream_t stream_;
  cudaEvent_t event_;

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct NvTrtRtxSyncStreamImpl : OrtSyncStreamImpl {
  NvTrtRtxSyncStreamImpl(cudaStream_t&& stream,
                         const OrtDevice& device,
                         AllocatorPtr cpu_allocator,
                         bool release_cpu_buffer_on_cuda_stream,
                         const OrtApi& ort_api_in)
      : stream_{
            stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, /*own*/ true,
            /*external_cudnn_handle*/ nullptr,
            /*external_cublas_handle*/ nullptr,
            // ep_info is used by GetResource which seems to be a somewhat ugly way to make arbitrary info that is
            // unrelated to the stream available to a custom op.
            // avoiding adding GetResource to OrtSyncStreamImpl as we should have a cleaner setup for custom ops,
            // so this argument value isn't used and doesn't matter.
            /*ep_info*/ CUDAExecutionProviderInfo{}},
        ort_api{ort_api_in} {
    ort_version_supported = ORT_API_VERSION;
    GetHandle = GetHandleImpl;
    CreateNotification = CreateNotificationImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
  }

  static void ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    delete static_cast<NvTrtRtxSyncStreamImpl*>(this_ptr);
  }

  static void* GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncStreamImpl*>(this_ptr);
    return impl.stream_.GetHandle();
  }

  static OrtStatus* CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                           _Outptr_ OrtSyncNotificationImpl** notification_impl) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncStreamImpl*>(this_ptr);
    *notification_impl = nullptr;

    std::unique_ptr<NvTrtRtxSyncNotificationImpl> notification;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(impl.stream_.GetHandle());

    RETURN_IF_ERROR(NvTrtRtxSyncNotificationImpl::Create(cuda_stream, impl.ort_api, notification));
    *notification_impl = notification.release();

    return nullptr;
  }

  static OrtStatus* FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncStreamImpl*>(this_ptr);
    impl.stream_.Flush();

    return nullptr;
  }

  static OrtStatus* OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
    auto& impl = *static_cast<NvTrtRtxSyncStreamImpl*>(this_ptr);
    RETURN_IF_STATUS_NOTOK(impl.stream_.CleanUpOnRunEnd());

    return nullptr;
  }

 private:
  // this is a little onion-ish as CudaStream is a onnxruntime::Stream and this is an OrtSyncStreamImpl that will be
  // used via plugin_ep::Stream, which is also an onnxruntime::Stream. in a 'real' plugin EP implementation
  // CudaStream would go away and the logic it has would be implemented directly here.
  CudaStream stream_;
  const OrtApi& ort_api;
};

#if defined(_WIN32)

// External Resource Import Implementation (D3D12 to CUDA)
/**
 * @brief Derived handle for imported external memory from D3D12 to CUDA.
 *
 * Derives from OrtExternalMemoryHandle (base struct) and adds CUDA-specific fields.
 * This struct holds the CUDA external memory object and the mapped device pointer
 * that can be used for zero-copy tensor creation.
 */
struct NvTrtRtxExternalMemoryHandle : OrtExternalMemoryHandle {
  CUexternalMemory ext_memory;  ///< CUDA external memory object
  CUdeviceptr mapped_ptr;       ///< Mapped device pointer for tensor access
  bool is_dedicated;            ///< Whether the D3D12 resource is a dedicated allocation

  NvTrtRtxExternalMemoryHandle(const OrtExternalMemoryDescriptor& descriptor_in)
      : ext_memory(nullptr), mapped_ptr(0), is_dedicated(true) {
    // Initialize base struct fields
    version = ORT_API_VERSION;
    descriptor = descriptor_in;
    ep_device = nullptr;
    Release = ReleaseCallback;
  }

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalMemoryHandle* handle) noexcept {
    if (handle == nullptr) return;
    auto derived = std::unique_ptr<NvTrtRtxExternalMemoryHandle>(
        static_cast<NvTrtRtxExternalMemoryHandle*>(handle));
    // Destroy the external memory object (also releases mapped buffer)
    if (derived->ext_memory != nullptr) {
      cuDestroyExternalMemory(derived->ext_memory);
    }
  }
};

/**
 * @brief Derived handle for imported external semaphore from D3D12 fence to CUDA.
 *
 * Derives from OrtExternalSemaphoreHandle (base struct) and adds CUDA-specific fields.
 * D3D12 timeline fences are imported as CUDA external semaphores, enabling
 * GPU-GPU synchronization between D3D12 and CUDA streams.
 */
struct NvTrtRtxExternalSemaphoreHandle : OrtExternalSemaphoreHandle {
  CUexternalSemaphore ext_semaphore;  ///< CUDA external semaphore object

  NvTrtRtxExternalSemaphoreHandle(const OrtExternalSemaphoreDescriptor& descriptor_in)
      : ext_semaphore(nullptr) {
    // Initialize base struct fields
    version = ORT_API_VERSION;
    descriptor = descriptor_in;
    ep_device = nullptr;
    Release = ReleaseCallback;
  }

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalSemaphoreHandle* handle) noexcept {
    if (handle == nullptr) return;
    auto derived = std::unique_ptr<NvTrtRtxExternalSemaphoreHandle>(
        static_cast<NvTrtRtxExternalSemaphoreHandle*>(handle));
    // Destroy the external semaphore object
    if (derived->ext_semaphore != nullptr) {
      cuDestroyExternalSemaphore(derived->ext_semaphore);
    }
  }
};

/**
 * @brief Implementation of OrtExternalResourceImporterImpl for NvTensorRtRtx EP.
 *
 * This struct implements the external resource importer interface using CUDA Driver APIs
 * to import D3D12 shared resources and timeline fences for zero-copy import.
 *
 * Supported handle types:
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE → CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP → CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
 * - ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE → CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
 */
struct NvTrtRtxExternalResourceImporterImpl : OrtExternalResourceImporterImpl {
  NvTrtRtxExternalResourceImporterImpl(const OrtEpDevice* ep_device, const OrtApi& ort_api_in)
      : ep_device_{ep_device}, ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
    ort_version_supported = ORT_API_VERSION;

    // Memory operations
    CanImportMemory = CanImportMemoryImpl;
    ImportMemory = ImportMemoryImpl;
    ReleaseMemory = ReleaseMemoryImpl;
    CreateTensorFromMemory = CreateTensorFromMemoryImpl;

    // Semaphore operations
    CanImportSemaphore = CanImportSemaphoreImpl;
    ImportSemaphore = ImportSemaphoreImpl;
    ReleaseSemaphore = ReleaseSemaphoreImpl;
    WaitSemaphore = WaitSemaphoreImpl;
    SignalSemaphore = SignalSemaphoreImpl;

    // Release
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanImportMemoryImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandleType handle_type) noexcept {
    (void)this_ptr;
    // CUDA supports both D3D12 resource and heap handles
    return handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE ||
           handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
  }

  static OrtStatus* ORT_API_CALL ImportMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryDescriptor* desc,
      _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept {
    auto& impl = *static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);

    if (desc == nullptr || out_handle == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to ImportMemory");
    }

    // Validate descriptor version - check minimum supported version for forward compatibility
    if (desc->version < ORT_API_VERSION) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "OrtExternalMemoryDescriptor version too old");
    }

    *out_handle = nullptr;

    // Validate handle type
    if (!CanImportMemoryImpl(this_ptr, desc->handle_type)) {
      return impl.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                       "Unsupported external memory handle type for CUDA import");
    }

    // Validate offset does not exceed allocation size
    if (desc->offset_bytes > desc->size_bytes) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "offset_bytes exceeds size_bytes in OrtExternalMemoryDescriptor");
    }

    // Set CUDA device for this EP. The imported external memory handle is associated with
    // the device where it was imported and remains valid regardless of subsequent cudaSetDevice
    // calls. Multi-GPU scenarios with different sessions/EPs work correctly because each
    // importer is bound to its EP's device via ep_device_->device_memory_info.
    ScopedContext ctx(impl.DeviceId());

    // Map ORT handle type to CUDA handle type
    CUexternalMemoryHandleType cu_handle_type;
    bool is_dedicated = true;
    switch (desc->handle_type) {
      case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:
        cu_handle_type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        is_dedicated = true;  // D3D12 committed resources are dedicated
        break;
      case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP:
        cu_handle_type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
        is_dedicated = false;  // D3D12 heaps are not dedicated
        break;
      default:
        // Should not reach here - CanImportMemory already validated handle type
        return impl.ort_api.CreateStatus(ORT_EP_FAIL, "Unexpected external memory handle type");
    }

    // Setup external memory handle descriptor
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC ext_mem_desc = {};
    ext_mem_desc.type = cu_handle_type;
    ext_mem_desc.handle.win32.handle = desc->native_handle;
    ext_mem_desc.size = desc->size_bytes;
    ext_mem_desc.flags = is_dedicated ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;

    // Import the external memory
    CUexternalMemory ext_memory = nullptr;
    CUresult cu_result = cuImportExternalMemory(&ext_memory, &ext_mem_desc);
    if (cu_result != CUDA_SUCCESS) {
      const char* error_str = nullptr;
      cuGetErrorString(cu_result, &error_str);
      std::string error_msg = "cuImportExternalMemory failed: ";
      error_msg += error_str ? error_str : "unknown error";
      return impl.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    // Map the external memory to get a device pointer
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc = {};
    buffer_desc.offset = desc->offset_bytes;
    buffer_desc.size = desc->size_bytes - desc->offset_bytes;
    buffer_desc.flags = 0;

    CUdeviceptr mapped_ptr = 0;
    cu_result = cuExternalMemoryGetMappedBuffer(&mapped_ptr, ext_memory, &buffer_desc);
    if (cu_result != CUDA_SUCCESS) {
      cuDestroyExternalMemory(ext_memory);
      const char* error_str = nullptr;
      cuGetErrorString(cu_result, &error_str);
      std::string error_msg = "cuExternalMemoryGetMappedBuffer failed: ";
      error_msg += error_str ? error_str : "unknown error";
      return impl.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    // Create and return the derived handle (cast to base pointer)
    OrtExternalMemoryDescriptor descriptor = {};  // make a copy for the handle
    descriptor.version = ORT_API_VERSION;
    descriptor.handle_type = desc->handle_type;
    descriptor.size_bytes = desc->size_bytes;
    descriptor.offset_bytes = desc->offset_bytes;
    auto handle = std::make_unique<NvTrtRtxExternalMemoryHandle>(descriptor);
    handle->ep_device = impl.ep_device_;
    handle->ext_memory = ext_memory;
    handle->mapped_ptr = mapped_ptr;
    handle->is_dedicated = is_dedicated;

    *out_handle = handle.release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandle* handle) noexcept {
    (void)this_ptr;

    if (handle == nullptr) {
      return;
    }

    // The handle has a Release callback that does the actual cleanup
    // This method is called from OrtExternalResourceImporterImpl::ReleaseMemory
    // The Release callback in the handle will call the static ReleaseCallback
    auto mem_handle = std::unique_ptr<NvTrtRtxExternalMemoryHandle>(
        static_cast<NvTrtRtxExternalMemoryHandle*>(handle));

    // Destroy the external memory object (also releases mapped buffer)
    if (mem_handle->ext_memory != nullptr) {
      cuDestroyExternalMemory(mem_handle->ext_memory);
    }
  }

  static OrtStatus* ORT_API_CALL CreateTensorFromMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryHandle* mem_handle,
      _In_ const OrtExternalTensorDescriptor* tensor_desc,
      _Outptr_ OrtValue** out_tensor) noexcept {
    auto& impl = *static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);

    if (mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to CreateTensorFromMemory");
    }

    // Validate descriptor version - check minimum supported version for forward compatibility
    if (tensor_desc->version < ORT_API_VERSION) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "OrtExternalTensorDescriptor version too old");
    }

    *out_tensor = nullptr;

    auto* handle = static_cast<const NvTrtRtxExternalMemoryHandle*>(mem_handle);

    // Validate tensor offset does not exceed available buffer size
    size_t available_size = handle->descriptor.size_bytes - handle->descriptor.offset_bytes;
    if (tensor_desc->offset_bytes > available_size) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "tensor offset_bytes exceeds available imported memory size");
    }

    // Calculate the data pointer with tensor offset
    void* data_ptr = reinterpret_cast<void*>(handle->mapped_ptr + tensor_desc->offset_bytes);

    // Get memory info from the EP device (the importer is associated with the OrtEpDevice)
    const OrtMemoryInfo* memory_info = impl.ep_device_->device_memory_info;

    // Create tensor that references the imported memory. The tensor does not own the memory -
    // the user manages the lifetime of both the OrtValue and OrtExternalMemoryHandle.
    // The user must keep the handle alive while the tensor is in use.
    // No deleter is needed since this is for inference inputs/outputs where the user controls lifetime.
    OrtStatus* status = impl.ort_api.CreateTensorWithDataAsOrtValue(
        memory_info,
        data_ptr,
        available_size - tensor_desc->offset_bytes,
        tensor_desc->shape,
        tensor_desc->rank,
        tensor_desc->element_type,
        out_tensor);

    return status;
  }

  static bool ORT_API_CALL CanImportSemaphoreImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreType type) noexcept {
    (void)this_ptr;
    // CUDA supports D3D12 timeline fences
    return type == ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  }

  static OrtStatus* ORT_API_CALL ImportSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalSemaphoreDescriptor* desc,
      _Outptr_ OrtExternalSemaphoreHandle** out_handle) noexcept {
    auto& impl = *static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);

    if (desc == nullptr || out_handle == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to ImportSemaphore");
    }

    // Validate descriptor version - check minimum supported version for forward compatibility
    if (desc->version < ORT_API_VERSION) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "OrtExternalSemaphoreDescriptor version too old");
    }

    *out_handle = nullptr;

    // Validate semaphore type
    if (!CanImportSemaphoreImpl(this_ptr, desc->type)) {
      return impl.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                       "Unsupported external semaphore type for CUDA import");
    }

    ScopedContext ctx(impl.DeviceId());

    // Setup external semaphore handle descriptor for D3D12 fence
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC ext_sem_desc = {};
    ext_sem_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    ext_sem_desc.handle.win32.handle = desc->native_handle;
    ext_sem_desc.flags = 0;

    // Import the external semaphore
    CUexternalSemaphore ext_semaphore = nullptr;
    CUresult cu_result = cuImportExternalSemaphore(&ext_semaphore, &ext_sem_desc);
    if (cu_result != CUDA_SUCCESS) {
      const char* error_str = nullptr;
      cuGetErrorString(cu_result, &error_str);
      std::string error_msg = "cuImportExternalSemaphore failed: ";
      error_msg += error_str ? error_str : "unknown error";
      return impl.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    // Create and return the derived handle (cast to base pointer)
    auto handle = std::make_unique<NvTrtRtxExternalSemaphoreHandle>(*desc);

    // Populate base struct fields
    handle->ep_device = impl.ep_device_;
    // Populate derived fields
    handle->ext_semaphore = ext_semaphore;

    *out_handle = handle.release();  // Return base pointer
    return nullptr;
  }

  static void ORT_API_CALL ReleaseSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle) noexcept {
    (void)this_ptr;

    if (handle == nullptr) {
      return;
    }

    auto sem_handle = std::unique_ptr<NvTrtRtxExternalSemaphoreHandle>(
        static_cast<NvTrtRtxExternalSemaphoreHandle*>(handle));

    if (sem_handle->ext_semaphore != nullptr) {
      cuDestroyExternalSemaphore(sem_handle->ext_semaphore);
    }
  }

  static OrtStatus* ORT_API_CALL WaitSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept {
    auto& impl = *static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);

    if (handle == nullptr || stream == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to WaitSemaphore");
    }

    auto* sem_handle = static_cast<NvTrtRtxExternalSemaphoreHandle*>(handle);

    // Get the CUDA stream from OrtSyncStream
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(impl.ort_api.SyncStream_GetHandle(stream));

    // Setup wait parameters for D3D12 fence (timeline semaphore)
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait_params = {};
    wait_params.params.fence.value = value;
    wait_params.flags = 0;

    // Wait on the external semaphore asynchronously
    CUresult cu_result = cuWaitExternalSemaphoresAsync(
        &sem_handle->ext_semaphore,
        &wait_params,
        1,  // numExtSems
        cuda_stream);

    if (cu_result != CUDA_SUCCESS) {
      const char* error_str = nullptr;
      cuGetErrorString(cu_result, &error_str);
      std::string error_msg = "cuWaitExternalSemaphoresAsync failed: ";
      error_msg += error_str ? error_str : "unknown error";
      return impl.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL SignalSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept {
    auto& impl = *static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);

    if (handle == nullptr || stream == nullptr) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to SignalSemaphore");
    }

    auto* sem_handle = static_cast<NvTrtRtxExternalSemaphoreHandle*>(handle);

    // Get the CUDA stream from OrtSyncStream
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(impl.ort_api.SyncStream_GetHandle(stream));

    // Setup signal parameters for D3D12 fence (timeline semaphore)
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal_params = {};
    signal_params.params.fence.value = value;
    signal_params.flags = 0;

    // Signal the external semaphore asynchronously
    CUresult cu_result = cuSignalExternalSemaphoresAsync(
        &sem_handle->ext_semaphore,
        &signal_params,
        1,  // numExtSems
        cuda_stream);

    if (cu_result != CUDA_SUCCESS) {
      const char* error_str = nullptr;
      cuGetErrorString(cu_result, &error_str);
      std::string error_msg = "cuSignalExternalSemaphoresAsync failed: ";
      error_msg += error_str ? error_str : "unknown error";
      return impl.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    return nullptr;
  }

  static void ORT_API_CALL ReleaseImpl(_In_ OrtExternalResourceImporterImpl* this_ptr) noexcept {
    delete static_cast<NvTrtRtxExternalResourceImporterImpl*>(this_ptr);
  }

  /// @brief Get the CUDA device ID from the EP device's memory info.
  int DeviceId() const {
    return ep_device_->device_memory_info->device.Id();
  }

 private:
  const OrtEpDevice* ep_device_;
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

#endif  // defined(_WIN32)

// OrtEpApi infrastructure to be able to use the NvTensorRTRTX EP as an OrtEpFactory for auto EP selection.
struct NvTensorRtRtxEpFactory : OrtEpFactory {
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

  NvTensorRtRtxEpFactory(const OrtApi& ort_api_in,
                         const OrtLogger& default_logger_in) : ort_api{ort_api_in},
                                                               ep_api{*ort_api_in.GetEpApi()},
                                                               default_logger{default_logger_in},
                                                               data_transfer_impl{ort_api_in} {
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
    ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
    CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;
    ort_version_supported = ORT_API_VERSION;  // Set to the ORT version we were compiled with.
  }

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const NvTensorRtRtxEpFactory*>(this_ptr);
    return factory->vendor_id;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return ORT_VERSION;
  }

  /**
   * @brief Checks if a given OrtHardwareDevice is a supported NVIDIA GPU.
   *
   * This function verifies if the provided hardware device corresponds to a physical
   * NVIDIA GPU that meets the minimum compute capability requirements for this execution provider.
   *
   * The check is performed by:
   * 1. Extracting the LUID (Locally Unique Identifier) from the device's metadata.
   * 2. Converting the string LUID to a 64-bit integer.
   * 3. Iterating through all available CUDA devices on the system.
   * 4. For each CUDA device, constructing its 64-bit LUID from its properties.
   * 5. Comparing the LUIDs. If a match is found, it checks if the device's
   *    compute capability is at least 8.0 (Ampere) or newer.
   *
   * @param device The OrtHardwareDevice to check.
   * @return True if the device is a supported NVIDIA GPU, false otherwise.
   */
  bool IsOrtHardwareDeviceSupported(const OrtHardwareDevice& device) {
#if _WIN32
    const auto& metadata_entries = device.metadata.Entries();
    const auto it = metadata_entries.find("LUID");
    if (it == metadata_entries.end()) {
      return false;
    }

    uint64_t target_luid;
    try {
      target_luid = std::stoull(it->second);
    } catch (const std::exception&) {
      return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
      return false;
    }

    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
        continue;
      }

      // The LUID is an 8-byte value, valid on Windows when luidDeviceNodeMask is non-zero.
      // We reconstruct the 64-bit integer representation from the raw bytes.
      if (prop.luidDeviceNodeMask == 0) {
        continue;
      }

      // Ensure the LUID is 8 bytes and reinterpret it directly as a uint64_t for comparison.
      static_assert(sizeof(prop.luid) == sizeof(uint64_t), "cudaDeviceProp::luid should be 8 bytes");
      uint64_t current_luid = *reinterpret_cast<const uint64_t*>(prop.luid);

      if (current_luid == target_luid) {
        // Ampere architecture or newer is required.
        return prop.major >= 8;
      }
    }

    return false;
#else
    const auto& metadata_entries = device.metadata.Entries();
    const auto it = metadata_entries.find("pci_bus_id");
    if (it == metadata_entries.end()) {
      return false;
    }
    auto& target_id = it->second;
    int cuda_device_idx = 0;
    if (cudaDeviceGetByPCIBusId(&cuda_device_idx, target_id.c_str()) != cudaSuccess) {
      return false;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, cuda_device_idx) != cudaSuccess) {
      return false;
    }
    // Ampere architecture or newer is required.
    return prop.major >= 8;
#endif
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of NvTensorRtRtx EP is not currently setup to partition a model among
  // multiple different NvTensorRtRtx backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
  // support one backend: gpu. To support a different backend, like npu, create a different factory instance
  // that only supports NPU.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<NvTensorRtRtxEpFactory*>(this_ptr);

    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    RETURN_IF_ERROR(factory->CreateMemoryInfoForDevices(num_cuda_devices));

    int16_t device_id = 0;
    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];

      if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id &&
          factory->IsOrtHardwareDeviceSupported(device)) {
        OrtKeyValuePairs* ep_options = nullptr;
        OrtKeyValuePairs* ep_metadata = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(device_id).c_str());

        RETURN_IF_ERROR(factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                    &ep_devices[num_ep_devices]));

        factory->ort_api.ReleaseKeyValuePairs(ep_options);
        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);

        const OrtMemoryInfo* gpu_mem_info = factory->gpu_memory_infos[device_id].get();
        const OrtMemoryInfo* host_accessible_mem_info = factory->host_accessible_memory_infos[device_id].get();

        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices], gpu_mem_info));
        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices], host_accessible_mem_info));
        num_ep_devices++;
        device_id++;
      }
    }
    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept {
    auto& factory = *static_cast<NvTensorRtRtxEpFactory*>(this_ptr);
    auto allocator_ = std::make_unique<NvTrtRtxOrtAllocator>(memory_info, factory.ort_api);
    *allocator = allocator_.release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
    delete static_cast<NvTrtRtxOrtAllocator*>(allocator);
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    auto& factory = *static_cast<NvTensorRtRtxEpFactory*>(this_ptr);
    *data_transfer = &factory.data_transfer_impl;
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return true;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* /*stream_options*/,
                                                               OrtSyncStreamImpl** ort_stream) noexcept {
    auto& factory = *static_cast<NvTensorRtRtxEpFactory*>(this_ptr);

    auto device_id = factory.ep_api.MemoryDevice_GetDeviceId(memory_device);
    cudaStream_t stream = nullptr;
    ScopedContext ctx(device_id);

    CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const OrtDevice* ort_device = static_cast<const OrtDevice*>(memory_device);

    auto impl = std::make_unique<NvTrtRtxSyncStreamImpl>(std::move(stream), *ort_device, nullptr,
                                                         /*release_cpu_buffer_on_cuda_stream*/ true,
                                                         factory.ort_api);
    *ort_stream = impl.release();
    return nullptr;
  }

  /** @brief Create an external resource importer for D3D12 to CUDA import.
   *
   * This enables zero-copy import of D3D12 shared resources and timeline fences.
   * The implementation uses CUDA Driver APIs (cuImportExternalMemory, cuImportExternalSemaphore).
   *
   * @param this_ptr The OrtEpFactory instance.
   * @param ep_device The OrtEpDevice to create the importer for.
   * @param out_importer Output parameter set to the created OrtExternalResourceImporterImpl.
   * @return nullptr on success, OrtStatus with error on failure.
   */
  static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* ep_device,
      OrtExternalResourceImporterImpl** out_importer) noexcept {
    auto& factory = *static_cast<NvTensorRtRtxEpFactory*>(this_ptr);

    if (out_importer == nullptr) {
      return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "out_importer cannot be nullptr");
    }

    *out_importer = nullptr;

#if defined(_WIN32)
    // Create the external resource importer
    auto importer = std::make_unique<NvTrtRtxExternalResourceImporterImpl>(ep_device, factory.ort_api);
    *out_importer = importer.release();

    return nullptr;
#else
    ORT_UNUSED_PARAMETER(ep_device);
    return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                        "External resource import is only available on Windows builds.");
#endif
  }

  /**
   * This function is called by the public C API GetModelCompatibilityForEpDevices.
   * It uses TensorRT RTX runtime directly to call runtime->getEngineValidity() to check the 64-byte engine header.
   *
   * @param this_ptr Factory instance pointer
   * @param devices Hardware devices (not used, validation is done against current system)
   * @param num_devices Number of devices
   * @param compatibility_info Hex-encoded 64-byte TensorRT RTX engine header (128 hex characters)
   * @param model_compatibility Output parameter for compatibility status
   * @return OrtStatus* nullptr on success, error status on failure
   */
  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      const char* compatibility_info,
      OrtCompiledModelCompatibility* model_compatibility) noexcept {
    auto& factory = *static_cast<NvTensorRtRtxEpFactory*>(this_ptr);

    // Validate input parameters
    if (compatibility_info == nullptr || model_compatibility == nullptr) {
      return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] Invalid arguments: compatibility_info or model_compatibility is null");
    }

    // Device parameters not used for header validation
    ORT_UNUSED_PARAMETER(devices);
    ORT_UNUSED_PARAMETER(num_devices);

    try {
      // If no compatibility info provided, validation not applicable
      if (compatibility_info[0] == '\0') {
        *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
        return nullptr;
      }

      // Decode hex string to binary
      std::vector<uint8_t> engine_header;
      try {
        engine_header = HexStringToBinary(std::string(compatibility_info));
      } catch (const std::exception& ex) {
        LOGS_DEFAULT(WARNING) << "[NvTensorRTRTX EP] Failed to decode engine header: " << ex.what();
        *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
        return nullptr;
      }

      // Validate header size (keep in sync with TensorRT engine header size)
      if (engine_header.size() != kTensorRTEngineHeaderSize) {
        LOGS_DEFAULT(WARNING) << "[NvTensorRTRTX EP] Invalid header size: " << engine_header.size()
                              << " bytes (expected " << kTensorRTEngineHeaderSize << ")";
        *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
        return nullptr;
      }

      // Create TensorRT runtime for validation
      static std::mutex runtime_creation_mutex;
      std::unique_ptr<nvinfer1::IRuntime> runtime;
      {
        std::lock_guard<std::mutex> lock(runtime_creation_mutex);
        TensorrtLogger& trt_logger = GetTensorrtLogger(false);
        runtime.reset(nvinfer1::createInferRuntime(trt_logger));
      }

      if (!runtime) {
        LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Failed to create TensorRT runtime";
        return factory.ort_api.CreateStatus(ORT_FAIL,
                                            "[NvTensorRTRTX EP] Failed to create TensorRT runtime");
      }

      // Use TensorRT's getEngineValidity to check compatibility
      uint64_t diagnostics = 0;
      nvinfer1::EngineValidity validity = runtime->getEngineValidity(
          engine_header.data(),
          engine_header.size(),
          &diagnostics);

      // Map TensorRT validity to ORT compatibility status
      switch (validity) {
        case nvinfer1::EngineValidity::kVALID:
          *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
          break;

        case nvinfer1::EngineValidity::kSUBOPTIMAL:
          LOGS_DEFAULT(WARNING) << "[NvTensorRTRTX EP] Engine compatible but recompilation recommended "
                                << "(diagnostics: 0x" << std::hex << diagnostics << std::dec << ")";
          *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
          break;

        case nvinfer1::EngineValidity::kINVALID:
          LOGS_DEFAULT(WARNING) << "[NvTensorRTRTX EP] Engine incompatible with this system "
                                << "(diagnostics: 0x" << std::hex << diagnostics << std::dec << ")";
          *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
          break;

        default:
          LOGS_DEFAULT(WARNING) << "[NvTensorRTRTX EP] Unknown validity status: "
                                << static_cast<int>(validity);
          *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
          break;
      }

      return nullptr;

    } catch (const std::exception& ex) {
      std::string error_msg = std::string("[NvTensorRTRTX EP] Exception during validation: ") + ex.what();
      LOGS_DEFAULT(ERROR) << error_msg;
      return factory.ort_api.CreateStatus(ORT_FAIL, error_msg.c_str());
    } catch (...) {
      LOGS_DEFAULT(ERROR) << "[NvTensorRTRTX EP] Unknown exception during validation";
      return factory.ort_api.CreateStatus(ORT_FAIL,
                                          "[NvTensorRTRTX EP] Unknown exception during validation");
    }
  }

  OrtStatus* CreateMemoryInfoForDevices(int num_devices) {
    gpu_memory_infos.reserve(num_devices);
    host_accessible_memory_infos.reserve(num_devices);

    for (int device_id = 0; device_id < num_devices; ++device_id) {
      OrtMemoryInfo* mem_info = nullptr;
      RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("NvTensorRTRTX", OrtMemoryInfoDeviceType_GPU,
                                                  /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                  /* device_id */ device_id,
                                                  OrtDeviceMemoryType_DEFAULT,
                                                  /*alignment*/ 0,
                                                  OrtAllocatorType::OrtDeviceAllocator,
                                                  &mem_info));
      gpu_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));

      mem_info = nullptr;
      RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("NvTensorRTRTX host accessible", OrtMemoryInfoDeviceType_GPU,
                                                  /*vendor*/ OrtDevice::VendorIds::NVIDIA,
                                                  /* device_id */ device_id,
                                                  OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                  /*alignment*/ 0,
                                                  OrtAllocatorType::OrtDeviceAllocator,
                                                  &mem_info));
      host_accessible_memory_infos.emplace_back(MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo));
    }
    return nullptr;
  }

 private:
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtLogger& default_logger;
  const std::string ep_name{kNvTensorRTRTXExecutionProvider};
  const std::string vendor{"NVIDIA"};

  // NVIDIA vendor ID. Refer to the ACPI ID registry (search NVIDIA): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id{0x10de};

  std::vector<MemoryInfoUniquePtr> gpu_memory_infos;
  std::vector<MemoryInfoUniquePtr> host_accessible_memory_infos;

  // we use a shared instance for the OrtDataTransferImpl instead of creating a new one on every call to
  NvTrtRtxDataTransferImpl data_transfer_impl;

  NvTensorRtRtxEpFactory(const NvTensorRtRtxEpFactory&) = delete;
  NvTensorRtRtxEpFactory& operator=(const NvTensorRtRtxEpFactory&) = delete;

  NvTensorRtRtxEpFactory(NvTensorRtRtxEpFactory&&) = default;
  NvTensorRtRtxEpFactory& operator=(NvTensorRtRtxEpFactory&&) = default;
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  // Factory could use registration_name or define its own EP name.
  auto factory = std::make_unique<NvTensorRtRtxEpFactory>(*ort_api, *default_logger);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<NvTensorRtRtxEpFactory*>(factory);
  return nullptr;
}
}
