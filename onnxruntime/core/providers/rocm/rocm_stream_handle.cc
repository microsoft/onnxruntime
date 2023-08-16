#include "core/providers/rocm/rocm_stream_handle.h"
#include "core/providers/rocm/rocm_common.h"
// #include "core/common/spin_pause.h"
#include "core/providers/rocm/rocm_resource.h"

namespace onnxruntime {

struct RocmNotification : public synchronize::Notification {
  RocmNotification(Stream& s) : Notification(s) {
    HIP_CALL_THROW(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
  }

  ~RocmNotification() {
    if (event_)
      HIP_CALL_THROW(hipEventDestroy(event_));
  }

  void Activate() override {
    // record event with hipEventBlockingSync so we can support sync on host without busy wait.
    HIP_CALL_THROW(hipEventRecord(event_, static_cast<hipStream_t>(stream_.GetHandle())));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.GetDevice().Type() == OrtDevice::GPU, "Unexpected device:", device_stream.GetDevice().ToString());
    // launch a wait command to the rocm stream
    HIP_CALL_THROW(hipStreamWaitEvent(static_cast<hipStream_t>(device_stream.GetHandle()), event_, 0));
  };

  void wait_on_host() {
    // CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    HIP_CALL_THROW(hipEventSynchronize(event_));
  }

  hipEvent_t event_;
};

RocmStream::RocmStream(hipStream_t stream,
                       const OrtDevice& device,
                       AllocatorPtr cpu_allocator,
                       bool release_cpu_buffer_on_rocm_stream,
                       bool own_flag,
                       miopenHandle_t external_miopen_handle,
                       rocblas_handle external_rocblas_handle) : Stream(stream, device),
                                                                 own_stream_(own_flag),
                                                                 cpu_allocator_(cpu_allocator),
                                                                 release_cpu_buffer_on_rocm_stream_(release_cpu_buffer_on_rocm_stream) {
  if (own_flag) {
    ROCBLAS_CALL_THROW(rocblas_create_handle(&rocblas_handle_));
    ROCBLAS_CALL_THROW(rocblas_set_stream(rocblas_handle_, stream));
    MIOPEN_CALL_THROW(miopenCreate(&miopen_handle_));
    MIOPEN_CALL_THROW(miopenSetStream(miopen_handle_, stream));
  } else {
    rocblas_handle_ = external_rocblas_handle;
    ROCBLAS_CALL_THROW(rocblas_set_stream(rocblas_handle_, stream));
    miopen_handle_ = external_miopen_handle;
    MIOPEN_CALL_THROW(miopenSetStream(miopen_handle_, stream));
  }
}

RocmStream::~RocmStream() {
  ORT_IGNORE_RETURN_VALUE(CleanUpOnRunEnd());
  if (own_stream_) {
    rocblas_destroy_handle(rocblas_handle_);
    miopenDestroy(miopen_handle_);
    auto* handle = GetHandle();
    if (handle)
      HIP_CALL_THROW(hipStreamDestroy(static_cast<hipStream_t>(handle)));
  }
}

std::unique_ptr<synchronize::Notification> RocmStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<RocmNotification>(*this);
}

void RocmStream::Flush() {
  if (own_stream_)
    HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(GetHandle())));
}

void RocmStream::EnqueDeferredCPUBuffer(void* cpu_buffer) {
  // stream is per thread, so don't need lock
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

struct CpuBuffersInfo {  // TODO: should be moved to base class
  AllocatorPtr allocator;
  std::unique_ptr<void*[]> buffers;
  // CPU buffer buffers[i].
  // Number of buffer points in "buffers".
  size_t n_buffers;
};

static void ReleaseCpuBufferCallback(hipStream_t /*stream*/, hipError_t /*status*/, void* raw_info) {  // TODO: should be moved to base class
  std::unique_ptr<CpuBuffersInfo> info = std::make_unique<CpuBuffersInfo>();
  info.reset(reinterpret_cast<CpuBuffersInfo*>(raw_info));
  for (size_t i = 0; i < info->n_buffers; ++i) {
    info->allocator->Free(info->buffers[i]);
  }
}

Status RocmStream::CleanUpOnRunEnd() {
  if (deferred_cpu_buffers_.empty())
    return Status::OK();
  // Release the ownership of cpu_buffers_info so that the underlying
  // object will keep alive until the end of ReleaseCpuBufferCallback.
  if (release_cpu_buffer_on_rocm_stream_ && cpu_allocator_->Info().alloc_type == OrtArenaAllocator) {
    std::unique_ptr<CpuBuffersInfo> cpu_buffers_info = std::make_unique<CpuBuffersInfo>();
    cpu_buffers_info->allocator = cpu_allocator_;
    cpu_buffers_info->buffers = std::make_unique<void*[]>(deferred_cpu_buffers_.size());
    for (size_t i = 0; i < deferred_cpu_buffers_.size(); ++i) {
      cpu_buffers_info->buffers[i] = deferred_cpu_buffers_.at(i);
    }
    cpu_buffers_info->n_buffers = deferred_cpu_buffers_.size();
    // TODO(wechi): CUDA deprecates cudaStreamAddCallback and
    // uses another API, cudaLaunchHostFunc(which can be
    // captured in CUDA graph). Once AMD adds similar feature,
    // we should replace the following line with
    //  hipLaunchHostFunc(stream, ReleaseCpuBufferCallback, cpu_buffers_info);

    // Release memory asynchronously to avoid blocking the compute stream.
    HIP_RETURN_IF_ERROR(hipStreamAddCallback(static_cast<hipStream_t>(GetHandle()), ReleaseCpuBufferCallback, cpu_buffers_info.release(), 0));
  } else {
    HIP_RETURN_IF_ERROR(hipStreamSynchronize(static_cast<hipStream_t>(GetHandle())));
    for (auto* buffer : deferred_cpu_buffers_) {
      cpu_allocator_->Free(buffer);
    }
  }

  deferred_cpu_buffers_.clear();
  return Status::OK();
}

void* RocmStream::GetResource(int version, int type) const {
  ORT_ENFORCE(version <= ORT_ROCM_RESOUCE_VERSION, "resource version unsupported!");
  void* resource{};
  switch (type) {
    case RocmResource::hip_stream_t:
      return reinterpret_cast<void*>(GetHandle());
      break;
    case RocmResource::miopen_handle_t:
      return reinterpret_cast<void*>(miopen_handle_);
      break;
    case RocmResource::rocblas_handle_t:
      return reinterpret_cast<void*>(rocblas_handle_);
      break;
    default:
      break;
  }
  return resource;
}

void WaitRocmNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  static_cast<RocmNotification*>(&notification)->wait_on_device(stream);
}

void WaitRocmNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  static_cast<RocmNotification*>(&notification)->wait_on_host();
}

void RegisterRocmStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_rocm_stream,
                               hipStream_t external_stream,
                               bool use_existing_stream,
                               miopenHandle_t external_miopen_handle,
                               rocblas_handle external_rocblas_handle) {
  // wait rocm notification on rocm ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitRocmNotificationOnDevice);
  // wait rocm notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitRocmNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator, release_cpu_buffer_on_rocm_stream](const OrtDevice& device) {
      hipStream_t stream = nullptr;
      HIP_CALL_THROW(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
      return std::make_unique<RocmStream>(stream, device, cpu_allocator, release_cpu_buffer_on_rocm_stream, true, nullptr, nullptr);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator,
                                                                release_cpu_buffer_on_rocm_stream,
                                                                external_stream,
                                                                external_miopen_handle,
                                                                external_rocblas_handle](const OrtDevice& device) {
      return std::make_unique<RocmStream>(external_stream, device, cpu_allocator, release_cpu_buffer_on_rocm_stream, false, external_miopen_handle, external_rocblas_handle);
    });
}

}  // namespace onnxruntime
