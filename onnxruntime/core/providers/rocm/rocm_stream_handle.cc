#include "core/providers/rocm/rocm_stream_handle.h"
#include "core/providers/rocm/rocm_common.h"
//#include "core/common/spin_pause.h"

namespace onnxruntime {

struct RocmNotification : public synchronize::Notification {
  RocmNotification(Stream* s) : Notification(s) {
    HIP_CALL_THROW(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
  }

  ~RocmNotification() {
    if (event_)
      HIP_CALL_THROW(hipEventDestroy(event_));
  }

  void Activate() override {
    // record event with hipEventBlockingSync so we can support sync on host with out busy wait.
    HIP_CALL_THROW(hipEventRecord(event_, static_cast<hipStream_t>(stream->handle)));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.device.Type() == OrtDevice::GPU);
    // launch a wait command to the cuda stream
    HIP_CALL_THROW(hipStreamWaitEvent(static_cast<hipStream_t>(device_stream.handle), event_, 0));
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
                       bool release_cpu_buffer_on_cuda_stream,
                       bool own_flag,
                       miopenHandle_t external_miopen_handle,
                       rocblas_handle external_rocblas_handle) : Stream(stream, device),
                                                                own_stream_(own_flag),
                                                                cpu_allocator_(cpu_allocator),
                                                                release_cpu_buffer_on_cuda_stream_(release_cpu_buffer_on_cuda_stream) {
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
    if (handle)
      HIP_CALL_THROW(hipStreamDestroy(static_cast<hipStream_t>(handle)));
  }
}

std::unique_ptr<synchronize::Notification> RocmStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<RocmNotification>(this);
}

void RocmStream::Flush() {
  // A temp fix: when use cuda graph, we can't flush it before cuda graph capture end
  // only flush when we own the stream (not external, not EP unified stream)
  if (own_stream_)
    HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(handle)));
}

void RocmStream::EnqueDeferredCPUBuffer(void* cpu_buffer) {
  // stream is per thread, so don't need lock
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

// TODO: release logic is in rocm_execution_provider
//struct CpuBuffersInfo {
//  // This struct stores the information needed
//  // to release CPU buffers allocated for GPU kernels.
//  // It's used to enqueue their release after
//  // associated GPU kernels in a CUDA stream.
//
//  // This is a CPU allocator in CUDA EP.
//  // It must be the one used to allocate the
//  // following pointers.
//  AllocatorPtr allocator;
//  // buffers[i] is the i-th pointer added by
//  // AddDeferredReleaseCPUPtr for a specific
//  // CUDA stream. For example, this fields
//  // should contain all values in
//  // deferred_release_buffer_pool_[my_stream]
//  // when release my_stream's buffers.
//  void** buffers;
//  // CPU buffer buffers[i].
//  // Number of buffer points in "buffers".
//  size_t n_buffers;
//};
//
//static void CUDART_CB ReleaseCpuBufferCallback(void* raw_info) {
//  auto info = reinterpret_cast<CpuBuffersInfo*>(raw_info);
//  // Uncomment the following line to check if all previous stream
//  // operations are done correctly.
//  // checkRocmErrors(tmp->status);
//  for (size_t i = 0; i < info->n_buffers; ++i) {
//    info->allocator->Free(info->buffers[i]);
//  }
//  delete[] info->buffers;
//  delete info;
//}

Status RocmStream::CleanUpOnRunEnd() {
//  if (deferred_cpu_buffers_.empty())
//    return Status::OK();
//  // Release the ownership of cpu_buffers_info so that the underlying
//  // object will keep alive until the end of ReleaseCpuBufferCallback.
//  if (release_cpu_buffer_on_cuda_stream_ && cpu_allocator_->Info().alloc_type == OrtArenaAllocator) {
//    auto cpu_buffers_info = new CpuBuffersInfo;
//    cpu_buffers_info->allocator = cpu_allocator_;
//    cpu_buffers_info->buffers = new void*[deferred_cpu_buffers_.size()];
//    for (size_t i = 0; i < deferred_cpu_buffers_.size(); ++i) {
//      cpu_buffers_info->buffers[i] = deferred_cpu_buffers_.at(i);
//    }
//    cpu_buffers_info->n_buffers = deferred_cpu_buffers_.size();
//    CUDA_RETURN_IF_ERROR(cudaLaunchHostFunc(static_cast<cudaStream_t>(handle), ReleaseCpuBufferCallback, cpu_buffers_info));
//  } else {
//    // for cuda graph case, if we launch the host function to cuda stream
//    // it seems be captured in cuda graph and replay, which cause wrong deletion.
//    // so in this mode, we manually sync the stream to make sure the copy is done
//    // then delete the buffers
//    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(handle)));
//    for (auto* buffer : deferred_cpu_buffers_) {
//      cpu_allocator_->Free(buffer);
//    }
//  }
//
//  deferred_cpu_buffers_.clear();
  return Status::OK();
}

// CPU Stream command handles
void WaitRocmNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  static_cast<RocmNotification*>(&notification)->wait_on_device(stream);
}

void WaitRocmNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  static_cast<RocmNotification*>(&notification)->wait_on_host();
}

void ReleaseCUdaNotification(void* handle) {
  delete static_cast<RocmNotification*>(handle);
}

void RegisterRocmStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_cuda_stream,
                               hipStream_t external_stream,
                               bool use_existing_stream,
                               miopenHandle_t external_miopen_handle,
                               rocblas_handle external_rocblas_handle) {
  // wait cuda notification on cuda ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitRocmNotificationOnDevice);
  // wait cuda notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitRocmNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator, release_cpu_buffer_on_cuda_stream](const OrtDevice& device) {
      hipStream_t stream = nullptr;
      HIP_CALL_THROW(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
      return std::make_unique<RocmStream>(stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, true, nullptr, nullptr);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator,
                                                                release_cpu_buffer_on_cuda_stream,
                                                                external_stream,
                                                                external_miopen_handle,
                                                                external_rocblas_handle](const OrtDevice& device) {
      return std::make_unique<RocmStream>(external_stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, false, external_miopen_handle, external_rocblas_handle);
    });
}

}  // namespace onnxruntime
