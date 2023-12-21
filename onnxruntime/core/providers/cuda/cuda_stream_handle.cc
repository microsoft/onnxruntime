// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cuda_resource.h"
#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

DeferredCpuAllocator::DeferredCpuAllocator(CudaStream& cuda_stream) : cuda_stream_(cuda_stream) {
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc =
      [](OrtAllocator* this_, size_t size) {
        auto self = reinterpret_cast<DeferredCpuAllocator*>(this_);
        return self->cuda_stream_.GetCpuAllocator()->Alloc(size);
      };
  OrtAllocator::Free =
      [](OrtAllocator* this_, void* p) {
        auto self = reinterpret_cast<DeferredCpuAllocator*>(this_);
        self->cuda_stream_.EnqueDeferredCPUBuffer(p);
      };
  OrtAllocator::Info =
      [](const OrtAllocator* this_) {
        auto self = reinterpret_cast<const DeferredCpuAllocator*>(this_);
        return &self->cuda_stream_.GetCpuAllocator()->Info();
      };
}

struct CudaNotification : public synchronize::Notification {
  CudaNotification(Stream& s) : Notification(s) {
    CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CudaNotification() {
    if (event_)
      CUDA_CALL_THROW(cudaEventDestroy(event_));
  }

  void Activate() override {
    // record event with cudaEventBlockingSync so we can support sync on host without busy wait.
    CUDA_CALL_THROW(cudaEventRecord(event_, static_cast<cudaStream_t>(stream_.GetHandle())));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.GetDevice().Type() == OrtDevice::GPU);
    // launch a wait command to the cuda stream
    CUDA_CALL_THROW(cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.GetHandle()),
                                        event_));
  };

  void wait_on_host() {
    // CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaEventSynchronize(event_));
  }

  cudaEvent_t event_;
};

CudaStream::CudaStream(cudaStream_t stream,
                       const OrtDevice& device,
                       AllocatorPtr cpu_allocator,
                       bool release_cpu_buffer_on_cuda_stream,
                       bool own_flag,
                       cudnnHandle_t external_cudnn_handle,
                       cublasHandle_t external_cublas_handle) : Stream(stream, device),
                                                                own_stream_(own_flag),
                                                                cpu_allocator_(cpu_allocator),
                                                                release_cpu_buffer_on_cuda_stream_(release_cpu_buffer_on_cuda_stream),
                                                                deferred_cpu_allocator_(*this) {
  if (own_flag) {
    CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, stream));
    CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));
    CUDNN_CALL_THROW(cudnnSetStream(cudnn_handle_, stream));
  } else {
    cublas_handle_ = external_cublas_handle;
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, stream));
    cudnn_handle_ = external_cudnn_handle;
    CUDNN_CALL_THROW(cudnnSetStream(cudnn_handle_, stream));
  }
}

CudaStream::~CudaStream() {
  ORT_IGNORE_RETURN_VALUE(CleanUpOnRunEnd());
  if (own_stream_) {
    cublasDestroy(cublas_handle_);
    cudnnDestroy(cudnn_handle_);
    auto* handle = GetHandle();
    if (handle)
      cudaStreamDestroy(static_cast<cudaStream_t>(handle));
  }
}

std::unique_ptr<synchronize::Notification> CudaStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<CudaNotification>(*this);
}

void CudaStream::Flush() {
  // A temp fix: when use cuda graph, we can't flush it before cuda graph capture end
  // only flush when we own the stream (not external, not EP unified stream)
  if (own_stream_)
    CUDA_CALL_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(GetHandle())));
}

void CudaStream::EnqueDeferredCPUBuffer(void* cpu_buffer) {
  // stream is per thread, so don't need lock
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

struct CpuBuffersInfo {
  // This struct stores the information needed
  // to release CPU buffers allocated for GPU kernels.
  // It's used to enqueue their release after
  // associated GPU kernels in a CUDA stream.

  // This is a CPU allocator in CUDA EP.
  // It must be the one used to allocate the
  // following pointers.
  AllocatorPtr allocator;
  // buffers[i] is the i-th pointer added by
  // AddDeferredReleaseCPUPtr for a specific
  // CUDA stream. For example, this fields
  // should contain all values in
  // deferred_release_buffer_pool_[my_stream]
  // when release my_stream's buffers.
  std::unique_ptr<void*[]> buffers;
  // CPU buffer buffers[i].
  // Number of buffer points in "buffers".
  size_t n_buffers;
};

static void CUDART_CB ReleaseCpuBufferCallback(void* raw_info) {
  std::unique_ptr<CpuBuffersInfo> info = std::make_unique<CpuBuffersInfo>();
  info.reset(reinterpret_cast<CpuBuffersInfo*>(raw_info));
  // Uncomment the following line to check if all previous stream
  // operations are done correctly.
  // checkCudaErrors(tmp->status);
  for (size_t i = 0; i < info->n_buffers; ++i) {
    info->allocator->Free(info->buffers[i]);
  }
}

Status CudaStream::CleanUpOnRunEnd() {
  if (deferred_cpu_buffers_.empty())
    return Status::OK();
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
}

void* CudaStream::GetResource(int version, int id) const {
  ORT_ENFORCE(version <= ORT_CUDA_RESOUCE_VERSION, "resource version unsupported!");
  void* resource{};
  switch (id) {
    case CudaResource::cuda_stream_t:
      return reinterpret_cast<void*>(GetHandle());
      break;
    case CudaResource::cudnn_handle_t:
      return reinterpret_cast<void*>(cudnn_handle_);
      break;
    case CudaResource::cublas_handle_t:
      return reinterpret_cast<void*>(cublas_handle_);
      break;
    case CudaResource::deferred_cpu_allocator_t:
      return const_cast<DeferredCpuAllocator*>(&deferred_cpu_allocator_);
      break;
    default:
      break;
  }
  return resource;
}

// CPU Stream command handles
void WaitCudaNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  static_cast<CudaNotification*>(&notification)->wait_on_device(stream);
}

void WaitCudaNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  static_cast<CudaNotification*>(&notification)->wait_on_host();
}

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_cuda_stream,
                               cudaStream_t external_stream,
                               bool use_existing_stream,
                               cudnnHandle_t external_cudnn_handle,
                               cublasHandle_t external_cublas_handle) {
  // wait cuda notification on cuda ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitCudaNotificationOnDevice);
  // wait cuda notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitCudaNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator, release_cpu_buffer_on_cuda_stream](const OrtDevice& device) {
      CUDA_CALL_THROW(cudaSetDevice(device.Id()));
      cudaStream_t stream = nullptr;
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      // CUDA_CALL_THROW(cudaStreamCreate(&stream));
      return std::make_unique<CudaStream>(stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, true, nullptr, nullptr);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator,
                                                                release_cpu_buffer_on_cuda_stream,
                                                                external_stream,
                                                                external_cudnn_handle,
                                                                external_cublas_handle](const OrtDevice& device) {
      return std::make_unique<CudaStream>(external_stream, device, cpu_allocator, release_cpu_buffer_on_cuda_stream, false, external_cudnn_handle, external_cublas_handle);
    });
}

}  // namespace onnxruntime
