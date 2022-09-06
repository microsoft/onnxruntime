#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {
struct CudaStreamBundle {
  cudaStream_t cuda_stream_{};
  cudnnHandle_t cudnn_handle_{};
  cublasHandle_t cublas_handle_{};
};

struct StreamPool {
  StreamPool(){};
  ~StreamPool() {
    for (const auto& s : streams_) {
      cudaStreamDestroy(s.cuda_stream_);
      cudnnDestroy(s.cudnn_handle_);
      cublasDestroy(s.cublas_handle_);
    }
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StreamPool);
  CudaStreamBundle GetStream() {
    if (streams_.empty()) {
      CudaStreamBundle stream_bundle;
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_bundle.cuda_stream_, cudaStreamNonBlocking));
      CUDNN_CALL_THROW(cudnnCreate(&stream_bundle.cudnn_handle_));
      CUDNN_CALL_THROW(cudnnSetStream(stream_bundle.cudnn_handle_, stream_bundle.cuda_stream_));
      CUBLAS_CALL_THROW(cublasCreate(&stream_bundle.cublas_handle_));
      CUBLAS_CALL_THROW(cublasSetStream(stream_bundle.cublas_handle_, stream_bundle.cuda_stream_));
      return stream_bundle;
    } else {
      CudaStreamBundle stream_bundle = streams_.back();
      streams_.pop_back();
      return stream_bundle;
    }
  }

  void PutStream(CudaStreamBundle stream) {
    streams_.push_back(stream);
  }
  std::vector<CudaStreamBundle> streams_;
};

StreamPool& GetStreamPool() {
  thread_local StreamPool stream_pool;
  return stream_pool;
}

struct CudaNotification : public synchronize::Notification {
  CudaNotification(Stream* s) : Notification(s) {
    CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CudaNotification() {
    if (event_)
      CUDA_CALL_THROW(cudaEventDestroy(event_));
  }

  void Activate() override {
    // record event with cudaEventBlockingSync so we can support sync on host with out busy wait.
    CUDA_CALL_THROW(cudaEventRecord(event_, static_cast<cudaStream_t>(stream->handle)));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.device.Type() == OrtDevice::GPU);
    // launch a wait command to the cuda stream
    CUDA_CALL_THROW(cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.handle),
                                        event_));
  };

  void wait_on_host() {
    // CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaEventSynchronize(event_));
  }

  cudaEvent_t event_;
};

CudaStream::CudaStream(cudaStream_t stream, const OrtDevice& device, bool own_flag,
                       cudnnHandle_t external_cudnn_handle, cublasHandle_t external_cublas_handle) : Stream(stream, device), own_stream_(own_flag) {
  cublas_handle_ = external_cublas_handle;
  cudnn_handle_ = external_cudnn_handle;
  if (!own_flag) {
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, stream));
    CUDNN_CALL_THROW(cudnnSetStream(cudnn_handle_, stream));
  }
}

CudaStream::~CudaStream() {
  if (own_stream_) {
    GetStreamPool().PutStream({static_cast<cudaStream_t>(handle), cudnn_handle_, cublas_handle_});
  }
}

std::unique_ptr<synchronize::Notification> CudaStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<CudaNotification>(this);
}

void CudaStream::Flush() {
  // A temp fix: when use cuda graph, we can't flush it before cuda graph capture end
  // only flush when we own the stream (not external, not EP unified stream)
  if (own_stream_)
    CUDA_CALL_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(handle)));
}

// CPU Stream command handles
void WaitCudaNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  static_cast<CudaNotification*>(&notification)->wait_on_device(stream);
}

void WaitCudaNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  static_cast<CudaNotification*>(&notification)->wait_on_host();
}

void ReleaseCUdaNotification(void* handle) {
  delete static_cast<CudaNotification*>(handle);
}

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                               const OrtDevice::DeviceType device_type,
                               cudaStream_t external_stream,
                               bool use_existing_stream,
                               cudnnHandle_t external_cudnn_handle,
                               cublasHandle_t external_cublas_handle) {
  // wait cuda notification on cuda ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitCudaNotificationOnDevice);
  // wait cuda notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitCudaNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [](const OrtDevice& device) {
      CudaStreamBundle stream_bundle = GetStreamPool().GetStream();
      return std::make_unique<CudaStream>(stream_bundle.cuda_stream_, device, true, stream_bundle.cudnn_handle_, stream_bundle.cublas_handle_);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [external_stream, external_cudnn_handle, external_cublas_handle](const OrtDevice& device) {
      return std::make_unique<CudaStream>(external_stream, device, false, external_cudnn_handle, external_cublas_handle);
    });
}

}  // namespace onnxruntime
