#include "onnxruntime_cxx_api.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "gsl/span"

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct CustomAllocator : OrtAllocator {
  CustomAllocator(const OrtMemoryInfo* mem_info) : memory_info{mem_info} {
    Alloc = AllocImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    Reserve = AllocImpl;  // no special reserve logic and most likely unnecessary unless you have your own arena
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* /*this_*/, size_t size) {
    // CustomAllocator& impl = *static_cast<CustomAllocator*>(this_);
    return malloc(size);
  }

  /// Free a block of memory previously allocated with OrtAllocator::Alloc
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* /*this_*/, void* p) {
    return free(p);
  }

  /// Return a pointer to an ::OrtMemoryInfo that describes this allocator
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const CustomAllocator& impl = *static_cast<const CustomAllocator*>(this_);
    return impl.memory_info;
  }

  const OrtMemoryInfo* memory_info;
};

//
// Class implementing Stream support for synchronization.
//
class StreamImpl : public OrtSyncStreamImpl, public ApiPtrs {
 public:
  StreamImpl(ApiPtrs apis);

  void* GetHandle() {
    return handle_;
  }

 private:
  static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ void* this_ptr, _In_ struct OrtSyncStream* stream,
                                                        _In_ size_t num_consumers,
                                                        _Outptr_ OrtSyncNotification** sync_notification) noexcept;

  // can maybe handle this at a higher level
  // as it's the same as the device->device wait fn
  // static OrtStatus* ORT_API_CALL GetWaitNotificationFuncImpl(_In_ void* this_ptr,
  //                                                           SyncWaitNotificationFn** notification_fn) noexcept {
  //}

  static OrtStatus* ORT_API_CALL FlushImpl(_In_ void* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ void* this_ptr) noexcept;

  static void* ORT_API_CALL GetResourceImpl(_In_ void* this_ptr, int32_t version, int32_t id) noexcept;

  // callback for EP library to release any internal state
  static void ORT_API_CALL ReleaseImpl(_In_ void* this_ptr) noexcept;

  void* handle_{nullptr};  // use the real stream type, like cudaStream_t or aclrtStream, etc.
};

//
// Class implementing synchronization notification support.
//
class NotificationImpl : public OrtSyncNotificationImpl, public ApiPtrs {
 public:
  NotificationImpl(ApiPtrs apis) : ApiPtrs(apis) {
    Activate = ActivateImpl;
    Release = ReleaseImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
  }

 private:
  static void ORT_API_CALL ActivateImpl(_In_ void* this_ptr) noexcept {
    auto& impl = *static_cast<NotificationImpl*>(this_ptr);
    static_cast<void>(impl);

    // CUDA: cudaEventRecord
    // CANN: aclrtRecordEvent
  }
  static void ORT_API_CALL WaitOnDeviceImpl(_In_ void* this_ptr, _In_ OrtSyncStream* stream) noexcept {
    auto& impl = *static_cast<NotificationImpl*>(this_ptr);
    StreamImpl& stream_impl = *static_cast<StreamImpl*>(impl.ep_api.SyncStream_GetStreamImpl(stream));
    static_cast<void>(stream_impl);

    // TODO: Setup the event or similar that will be activated on notification.
    // See CudaNotification or CannNotification for examples
    //
    // e.g.
    // CUDA: cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.GetHandle()), event_)
    // CANN: aclrtStreamWaitEvent(static_cast<aclrtStream>(device_stream.GetHandle()), event_)
    //
    // `event_` should be a member that is created in the ctor.
    // The stream handle should come from the StreamImpl instance and can be the real type so no static_cast is needed.
  }

  static void ORT_API_CALL WaitOnHostImpl(_In_ void* this_ptr) noexcept {
    auto& impl = *static_cast<NotificationImpl*>(this_ptr);
    static_cast<void>(impl);

    // CUDA: cudaEventSynchronize(event_)
    // CANN: aclrtSynchronizeEvent(event_)
  }

  static void ORT_API_CALL ReleaseImpl(_In_ void* this_ptr) noexcept {
    delete static_cast<NotificationImpl*>(this_ptr);
  }

  void* event_{NULL};  // placeholder. e.g. CANN uses aclrtEvent, CUDA uses cudaEvent_t
};

//
// StreamImpl implementation
//

StreamImpl::StreamImpl(ApiPtrs apis) : ApiPtrs(apis) {
  version = ORT_API_VERSION;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  GetResource = GetResourceImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::CreateNotificationImpl(_In_ void* this_ptr, _In_ struct OrtSyncStream* stream,
                                                           _In_ size_t /*num_consumers*/,
                                                           _Outptr_ OrtSyncNotification** sync_notification) noexcept {
  auto& impl = *static_cast<StreamImpl*>(this_ptr);
  auto notification = std::make_unique<NotificationImpl>(impl);
  auto* status = impl.ep_api.CreateSyncNotification(stream, notification.get(), sync_notification);

  if (status != nullptr) {
    return status;  // error occurred
  }

  notification.release();
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::FlushImpl(_In_ void* /*this_ptr*/) noexcept {
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL StreamImpl::OnSessionRunEndImpl(_In_ void* /*this_ptr*/) noexcept {
  return nullptr;
}

/*static*/  // TODO: Is this required?
void* ORT_API_CALL StreamImpl::GetResourceImpl(_In_ void* /*this_ptr*/, int32_t /*version*/, int32_t /*id*/) noexcept {
  return nullptr;
}

// callback for EP library to release any internal state
/*static*/
void ORT_API_CALL StreamImpl::ReleaseImpl(_In_ void* this_ptr) noexcept {
  auto* impl = static_cast<StreamImpl*>(this_ptr);
  delete impl;
}

class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, session_options_{session_options}, logger_{logger} {
    // Initialize the execution provider's function table
    GetName = GetNameImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void)status;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

 private:
  static const char* GetNameImpl(const OrtEp* this_ptr) {
    const auto* ep = static_cast<const ExampleEp*>(this_ptr);
    return ep->name_.c_str();
  }

  static OrtStatus* CreateSyncStreamForDeviceImpl(OrtEp* this_ptr, /*const OrtSession* session,*/
                                                  const OrtMemoryDevice* memory_device,
                                                  OrtSyncStream** stream) {
    auto& ep = *static_cast<ExampleEp*>(this_ptr);

    auto sync_stream = std::make_unique<StreamImpl>(ep);
    return ep.ep_api.CreateSyncStream(memory_device, sync_stream.get(), stream);
  }

  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};

struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    ReleaseDataTransfer = ReleaseDataTransferImpl;

    // setup the OrtMemoryInfo instances required by the EP.

    // for the sake of this example we specify a CPU allocator with no arena and 1K alignment (arbitrary)
    OrtMemoryInfo* mem_info = nullptr;
    auto* status = ort_api.CreateMemoryInfo_V2("ExampleEP CPU", OrtMemoryInfoDeviceType_CPU,
                                               /*vendor*/ 0x0000, /* device_id */ 0,
                                               OrtMemType::OrtMemTypeDefault,
                                               /*alignment*/ 1024,
                                               OrtAllocatorType::OrtDeviceAllocator,  // no arena
                                               &mem_info);
    assert(status == nullptr);  // should never fail.

    cpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

    //
    // GPU allocator OrtMemoryInfo for example purposes
    mem_info = nullptr;
    status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU", OrtMemoryInfoDeviceType_GPU,
                                         /*vendor*/ 0x0000, /* device_id */ 0,
                                         OrtMemType::OrtMemTypeDefault,
                                         /*alignment*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator,
                                         &mem_info);
    assert(status == nullptr);  // should never fail.
    default_gpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

    mem_info = nullptr;
    status = ort_api.CreateMemoryInfo_V2("ExampleEP GPU pinned", OrtMemoryInfoDeviceType_CPU,
                                         /*vendor*/ 0x0000, /* device_id */ 0,
                                         OrtMemType::OrtMemTypeCPU,
                                         /*alignment*/ 0,
                                         OrtAllocatorType::OrtDeviceAllocator,
                                         &mem_info);
    assert(status == nullptr);  // should never fail.
    pinned_gpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) {
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
        auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                   &ep_devices[num_ep_devices++]);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        if (status != nullptr) {
          return status;
        }
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

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) {
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

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
    delete dummy_ep;
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEpFactory* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _In_ const OrtKeyValuePairs* /*allocator_options*/,
                                                     _Outptr_ OrtAllocator** allocator) noexcept {
    *allocator = nullptr;
    auto& factory = *static_cast<ExampleEpFactory*>(this_ptr);

    // TODO: If we want to support usage of BFCArena the OrtEnv will have to wrap the OrtAllocator returned with
    // IAllocatorImplWrappingOrtAllocator (to get an IAllocator), put that in std::unique_ptr, and create the BFCArena.
    // Any options specific to the arena would need to be parsed/applied at that level.
    // We also need to wire the OrtEpFactory::ReleaseAllocator call into the IAllocatorImplWrappingOrtAllocator dtor.

    if (memory_info == factory.cpu_memory_info_.get()) {
      // create a CPU allocator. use the basic OrtAllocator for this example. in real code you could derive a class
      // from it
      auto cpu_allocator = std::make_unique<CustomAllocator>(memory_info);
      *allocator = cpu_allocator.release();
    } else if (memory_info == factory.default_gpu_memory_info_.get()) {
      // create a GPU allocator
      return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "Example is not implemented.");
    } else if (memory_info == factory.pinned_gpu_memory_info_.get()) {
      // create a pinned memory allocator
      return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "Example is not implemented.");
    } else {
      return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "Unknown memory info provided to CreateAllocator.");
    }

    return nullptr;
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(_In_ OrtEpFactory* /*this*/, _In_ OrtAllocator* allocator) noexcept {
    delete static_cast<CustomAllocator*>(allocator);
  }

  struct ExampleDataTransfer : OrtDataTransfer, ApiPtrs {
    ExampleDataTransfer(ApiPtrs api_ptrs,
                        const OrtMemoryDevice* device_mem_info_,
                        const OrtMemoryDevice* shared_mem_info_ = nullptr)
        : ApiPtrs(api_ptrs), device_mem_info{device_mem_info_}, shared_mem_info{shared_mem_info_} {
      CanCopy = CanCopyImpl;
      CopyTensors = CopyTensorsImpl;
    }

    static bool ORT_API_CALL CanCopyImpl(_In_ void* this_ptr,
                                         _In_ const OrtMemoryDevice* src_memory_device,
                                         _In_ const OrtMemoryDevice* dst_memory_device) noexcept {
      auto& impl = *static_cast<ExampleDataTransfer*>(this_ptr);
      bool src_is_our_device = impl.ep_api.OrtMemoryDevice_AreEqual(src_memory_device, impl.device_mem_info);
      bool dst_is_our_device = impl.ep_api.OrtMemoryDevice_AreEqual(dst_memory_device, impl.device_mem_info);

      return src_is_our_device || dst_is_our_device;
    }

    // function to copy one or more tensors.
    // implementation can optionally use async copy if a stream is available for the input.
    static OrtStatus* ORT_API_CALL CopyTensorsImpl(_In_ void* this_ptr,
                                                   _In_reads_(num_tensors) const OrtValue** src_tensors_ptr,
                                                   _In_reads_(num_tensors) OrtValue** dst_tensors_ptr,
                                                   _In_reads_(num_tensors) OrtSyncStream** streams_ptr,
                                                   _In_ size_t num_tensors) noexcept {
      auto& impl = *static_cast<ExampleDataTransfer*>(this_ptr);

      auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
      auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);
      auto streams = gsl::make_span<OrtSyncStream*>(streams_ptr, num_tensors);

      for (size_t i = 0; i < num_tensors; ++i) {
        auto* sync_stream = streams[i];
        StreamImpl* stream = nullptr;
        if (sync_stream) {
          // get the Stream implementation which has the handle for managing async copies
          stream = static_cast<StreamImpl*>(impl.ep_api.SyncStream_GetStreamImpl(sync_stream));
        }

        const OrtMemoryDevice* src_device = nullptr;
        const OrtMemoryDevice* dst_device = nullptr;
        auto* status = impl.ep_api.OrtValue_GetMemoryDevice(src_tensors[i], &src_device);
        if (status != nullptr) {
          return status;
        }

        status = impl.ep_api.OrtValue_GetMemoryDevice(dst_tensors[i], &dst_device);
        if (status != nullptr) {
          return status;
        }

        auto src_device_type = impl.ep_api.OrtMemoryDevice_GetDeviceType(src_device);
        auto dst_device_type = impl.ep_api.OrtMemoryDevice_GetDeviceType(dst_device);
        auto src_mem_type = impl.ep_api.OrtMemoryDevice_GetMemoryType(src_device);
        auto dst_mem_type = impl.ep_api.OrtMemoryDevice_GetMemoryType(dst_device);

        bool copy_involves_pinned_memory = src_mem_type == OrtMemType::OrtMemTypeCPU ||
                                           dst_mem_type == OrtMemType::OrtMemTypeCPU;

        if (dst_device_type == OrtMemoryInfoDeviceType_GPU) {
          if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
            // GPU -> GPU
          } else {
            // CPU -> GPU
          }
        } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
          // GPU -> CPU
        } else {
          // CPU -> CPU involves copy to/from pinned memory and a synchronize may be required first
          assert(copy_involves_pinned_memory);
        }
      }
    }

   private:
    const OrtMemoryDevice* device_mem_info;
    const OrtMemoryDevice* shared_mem_info;
  };

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(_In_ OrtEpFactory* /*this_ptr*/,
                                                        _Outptr_ OrtDataTransfer** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
  }

  static void ORT_API_CALL ReleaseDataTransferImpl(_In_ OrtEpFactory* /*this_ptr*/,
                                                   _In_ OrtDataTransfer* /*data_transfer*/) noexcept {
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr cpu_memory_info_;

  // for example purposes. if the EP used GPU, and pinned/shared memory was required for data transfer, these are the
  // OrtMemoryInfo instance required for that.
  MemoryInfoUniquePtr default_gpu_memory_info_;
  MemoryInfoUniquePtr pinned_gpu_memory_info_;
};

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<ExampleEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api, *ep_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<ExampleEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
