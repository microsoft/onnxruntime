// Copyright (c) Microsoft Corporation. All rights reserved.

#include "core/providers/cuda/plugin_ep/sync_stream.h"

namespace cuda_plugin_ep {
namespace {
struct CpuBuffersInfo {
  // This struct stores the information needed to release CPU buffers allocated for GPU kernels.
  // It's used to enqueue their release after associated GPU kernels in a CUDA stream.

  // This is a CPU allocator in CUDA EP. It must be the one used to allocate the following pointers.
  OrtAllocator* allocator;

  // buffers[i] is the i-th pointer added by AddDeferredReleaseCPUPtr for a specific CUDA stream.
  // For example, this field should contain all values in deferred_release_buffer_pool_[my_stream]
  // when releasing my_stream's buffers.
  std::unique_ptr<void*[]> buffers;

  // Number of buffer points in "buffers".
  size_t n_buffers;
};

void CUDART_CB ReleaseCpuBufferCallback(void* raw_info) {
  std::unique_ptr<CpuBuffersInfo> info(static_cast<CpuBuffersInfo*>(raw_info));

  // Uncomment to check if all previous stream operations were done correctly.
  // checkCudaErrors(tmp->status);

  for (size_t i = 0; i < info->n_buffers; ++i) {
    info->allocator->Free(info->allocator, info->buffers[i]);
  }
}
}  // namespace

CudaSyncNotificationImpl::CudaSyncNotificationImpl(cudaStream_t stream) : stream_{stream} {
  ort_version_supported = ORT_API_VERSION;
  Activate = ActivateImpl;
  WaitOnDevice = WaitOnDeviceImpl;
  WaitOnHost = WaitOnHostImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* CudaSyncNotificationImpl::Create(cudaStream_t stream,
                                            std::unique_ptr<CudaSyncNotificationImpl>& notification) {
  notification.reset(new CudaSyncNotificationImpl(stream));  // can't use make_unique with private ctor
  CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&notification->event_, cudaEventDisableTiming));

  return nullptr;
}

/*static*/
OrtStatus* CudaSyncNotificationImpl::ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventRecord(impl.event_, impl.stream_));

  return nullptr;
}

/*static*/
OrtStatus* CudaSyncNotificationImpl::WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                      _In_ OrtSyncStream* consumer_stream) noexcept {
  auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);

  // setup the consumer stream to wait on our event.
  void* consumer_handle = Shared::ort_api->SyncStream_GetHandle(consumer_stream);
  CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_handle), impl.event_));

  return nullptr;
}

/*static*/
OrtStatus* CudaSyncNotificationImpl::WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept {
  auto& impl = *static_cast<CudaSyncNotificationImpl*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(impl.event_));

  return nullptr;
}

CudaSyncNotificationImpl::~CudaSyncNotificationImpl() {
  cudaEventDestroy(event_);
}

CudaSyncStreamImpl::CudaSyncStreamImpl(cudaStream_t&& stream, const OrtMemoryDevice& device)
    : stream_{stream} {
  ort_version_supported = ORT_API_VERSION;
  GetHandle = GetHandleImpl;
  CreateNotification = CreateNotificationImpl;
  Flush = FlushImpl;
  OnSessionRunEnd = OnSessionRunEndImpl;
  Release = ReleaseImpl;
}

OrtStatus* CudaSyncStreamImpl::CreateHandles() {
  CUBLAS_RETURN_IF_ERROR(cublasCreate(&cublas_handle_));
  CUBLAS_RETURN_IF_ERROR(cublasSetStream(cublas_handle_, stream_));
  CUDNN_RETURN_IF_ERROR(cudnnCreate(&cudnn_handle_));
  CUDNN_RETURN_IF_ERROR(cudnnSetStream(cudnn_handle_, stream_));
}

/*static*/
OrtStatus* CudaSyncStreamImpl::Create(cudaStream_t&& stream,
                                      const OrtMemoryDevice& device,
                                      std::unique_ptr<CudaSyncStreamImpl>& sync_stream) {
  // private ctor so can't use make_unique
  std::unique_ptr<CudaSyncStreamImpl> impl(new CudaSyncStreamImpl(std::move(stream), device));
  RETURN_IF_ERROR(impl->CreateHandles());
  sync_stream = std::move(impl);
  return nullptr;
}

/*static*/
OrtStatus* CudaSyncStreamImpl::CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                      _Outptr_ OrtSyncNotificationImpl** notification_impl) noexcept {
  auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
  *notification_impl = nullptr;

  std::unique_ptr<CudaSyncNotificationImpl> notification;

  RETURN_IF_ERROR(CudaSyncNotificationImpl::Create(impl.stream_, notification));
  *notification_impl = notification.release();

  return nullptr;
}

/*static*/ OrtStatus* CudaSyncStreamImpl::FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(impl.stream_)));

  return nullptr;
}

/*static*/ OrtStatus* CudaSyncStreamImpl::OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto& impl = *static_cast<CudaSyncStreamImpl*>(this_ptr);

  // release_cpu_buffer_on_cuda_stream_ = !graph capture enabled for session

  /*
  if (deferred_cpu_buffers_.empty()) {
    return Status::OK();
  }

  // Release the ownership of cpu_buffers_info so that the underlying
  // object will keep alive until the end of ReleaseCpuBufferCallback.
  if (release_cpu_buffer_on_cuda_stream_ && cpu_allocator_->Info().alloc_type == OrtArenaAllocator) {
    std::unique_ptr<CpuBuffersInfo> cpu_buffers_info = std::make_unique<CpuBuffersInfo>();
    cpu_buffers_info->allocator = cpu_allocator_;
    cpu_buffers_info->buffers = std::make_unique<void*[]>(deferred_cpu_buffers_.size());
    for (size_t i = 0; i < deferred_cpu_buffers_.size(); ++i) {
      cpu_buffers_info->buffers[i] = deferred_cpu_buffers_.at(i);
    }
    cpu_buffers_info->n_buffers = deferred_cpu_buffers_.size();
    CUDA_RETURN_IF_ERROR(cudaLaunchHostFunc(static_cast<cudaStream_t>(GetHandle()), ReleaseCpuBufferCallback, cpu_buffers_info.release()));
  } else {
    // for cuda graph case, if we launch the host function to cuda stream
    // it seems be captured in cuda graph and replay, which cause wrong deletion.
    // so in this mode, we manually sync the stream to make sure the copy is done
    // then delete the buffers
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(GetHandle())));
    for (auto* buffer : deferred_cpu_buffers_) {
      cpu_allocator_->Free(buffer);
    }
  }

  deferred_cpu_buffers_.clear();
  return Status::OK();
  */

  return nullptr;
}

}  // namespace cuda_plugin_ep
