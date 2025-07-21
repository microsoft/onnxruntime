// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "nv_provider_factory.h"
#include <atomic>
#include "nv_execution_provider.h"
#include "nv_provider_factory_creator.h"
#include "nv_data_transfer.h"
#include "nv_allocator.h"
#include "core/framework/provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "core/providers/nv_tensorrt_rtx/nv_execution_provider_custom_ops.h"
#include <string.h>
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/cuda_stream_handle.h"

using namespace onnxruntime;

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

      const OrtMemoryDevice *src_device = nullptr, *dst_device = nullptr;
      RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(src_tensor, &src_device));
      RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(dst_tensor, &dst_device));

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

  cudaStream_t& stream_;
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
    cudaStream_t* cuda_stream = static_cast<cudaStream_t*>(impl.stream_.GetHandle());

    RETURN_IF_ERROR(NvTrtRtxSyncNotificationImpl::Create(*cuda_stream, impl.ort_api, notification));
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


// OrtEpApi infrastructure to be able to use the NvTensorRTRTX EP as an OrtEpFactory for auto EP selection.
struct NvTensorRtRtxEpFactory : OrtEpFactory {

  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;


  NvTensorRtRtxEpFactory(const OrtApi& ort_api_in,
                         const OrtLogger& default_logger_in): ort_api{ort_api_in},
                         ep_api{*ort_api_in.GetEpApi()},
                         default_logger{default_logger_in},
                         data_transfer_impl{ort_api_in}
                         {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetVendorId = GetVendorIdImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;

    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;

    CreateDataTransfer = CreateDataTransferImpl;

    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
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
          factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(device_id).c_str());

        OrtEpDevice* ep_device = nullptr;
        RETURN_IF_ERROR(factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, ep_options,
                                                                   &ep_devices[num_ep_devices++]));
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        const OrtMemoryInfo* gpu_mem_info = factory->gpu_memory_infos[device_id].get();
        const OrtMemoryInfo* host_accessible_mem_info = factory->host_accessible_memory_infos[device_id].get();

        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, gpu_mem_info));
        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, host_accessible_mem_info));
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
    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
    CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const OrtDevice* ort_device = static_cast<const OrtDevice*>(memory_device);

    auto impl = std::make_unique<NvTrtRtxSyncStreamImpl>(std::move(stream), *ort_device, nullptr,
                                                     /*release_cpu_buffer_on_cuda_stream*/ true,
                                                     factory.ort_api);
    *ort_stream = impl.release();
    return nullptr;
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
<<<<<<< HEAD
  auto factory_gpu = std::make_unique<NvTensorRtRtxEpFactory>(*ort_api, *default_logger, OrtHardwareDeviceType_GPU);
=======
  auto factory_gpu = std::make_unique<NvTensorRtRtxEpFactory>(*ort_api, *default_logger);
>>>>>>> c8d880128 (!. copied changes from CUDA EP for the EP afctory interface.)

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory_gpu.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<NvTensorRtRtxEpFactory*>(factory);
  return nullptr;
}
}
