#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

struct EventPool {
  ~EventPool() {
    for (const auto& e : events_) {
      CUDA_CALL_THROW(cudaEventDestroy(e));
    }
    events_.clear();
  }
  cudaEvent_t GetEvent() {
    if (events_.empty()) {
      cudaEvent_t event;
      CUDA_CALL_THROW(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      return event;
    } else {
      cudaEvent_t event = events_.back();
      events_.pop_back();
      return event;
    }
  }

  void PutEvent(cudaEvent_t event) {
    ORT_ENFORCE(event);
    events_.push_back(event);
  }

  std::vector<cudaEvent_t> events_;
};

struct EventPool& GetEventPool() {
  thread_local EventPool event_pool;
  return event_pool;
}


struct PerThreadHandles {
  cudnnHandle_t cudnn_handle_{};
  cudaStream_t cudnn_stream_{};

  cublasHandle_t cublas_handle_{};
  cudaStream_t cublas_stream_{};

  cudnnHandle_t GetCudnnHandle(cudaStream_t stream) {
    if (!cudnn_handle_) {
      CUDNN_CALL(cudnnCreate(&cudnn_handle_));
    }
    if (cudnn_stream_ != stream) {
      cudnn_stream_ = stream;
      CUDNN_CALL(cudnnSetStream(cudnn_handle_, cudnn_stream_));
    }
    return cudnn_handle_;
  }
  cublasHandle_t GetCublasHandle(cudaStream_t stream) {
    if (!cublas_handle_) {
      CUBLAS_CALL(cublasCreate(&cublas_handle_)); 
    } 
    if (cublas_stream_ != stream) {
      cublas_stream_ = stream;
      CUBLAS_CALL(cublasSetStream(cublas_handle_, cublas_stream_));
    }
    return cublas_handle_;
  }
};

PerThreadHandles& GetHandles() {
  thread_local PerThreadHandles handles;
  return handles;
}

struct CudaNotification : public synchronize::Notification {
  CudaNotification(Stream* s) : Notification(s) {
    //CUDA_CALL_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    event_ = GetEventPool().GetEvent();
  }
  
  ~CudaNotification() {

    //GetHandles().ReleaseCudnnHandle(static_cast<cudaStream_t>(stream->handle));
    //GetHandles().ReleaseCublasHandle(static_cast<cudaStream_t>(stream->handle));

    if (event_)
      GetEventPool().PutEvent(event_);
      //CUDA_CALL_THROW(cudaEventDestroy(event_));
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
    
    //CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaEventSynchronize(event_));
  }

  cudaEvent_t event_;
};

CudaStream::CudaStream(cudaStream_t stream, const OrtDevice& device, bool own_flag,
                       cudnnHandle_t external_cudnn_handle, cublasHandle_t external_cublas_handle) : Stream(stream, device), own_stream_(own_flag) {
  //if (own_flag) {
  //  CUBLAS_CALL(cublasCreate(&cublas_handle_));
  //  CUBLAS_CALL(cublasSetStream(cublas_handle_, stream));
  //  CUDNN_CALL(cudnnCreate(&cudnn_handle_));
  //  CUDNN_CALL(cudnnSetStream(cudnn_handle_, stream));
  //} else {
  //  cublas_handle_ = external_cublas_handle;
  //  CUBLAS_CALL(cublasSetStream(cublas_handle_, stream));
  //  cudnn_handle_ = external_cudnn_handle;
  //  CUDNN_CALL(cudnnSetStream(cudnn_handle_, stream));
  //}
}

CudaStream::~CudaStream() {
  if (own_stream_) {
    //CUBLAS_CALL(cublasDestroy(cublas_handle_));
    //CUDNN_CALL(cudnnDestroy(cudnn_handle_));
    if (handle) {
      CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(handle)));
    }
  }
}

std::unique_ptr<synchronize::Notification> CudaStream::CreateNotification(size_t /*num_consumers*/){
  return std::make_unique<CudaNotification>(this);
}

void CudaStream::Flush(){
  // A temp fix: when use cuda graph, we can't flush it before cuda graph capture end
  // only flush when we own the stream (not external, not EP unified stream)
  if (own_stream_)
    CUDA_CALL_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(handle))); 
}

cudnnHandle_t CudaStream::GetCudnnHandle() {
  return GetHandles().GetCudnnHandle(static_cast<cudaStream_t>(handle));
  //return cudnn_handle_;
}

cublasHandle_t CudaStream::GetCublasHandle() {
  return GetHandles().GetCublasHandle(static_cast<cudaStream_t>(handle));
  //return cublas_handle_;
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
      cudaStream_t stream = nullptr;
      //CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CALL_THROW(cudaStreamCreate(&stream));
      return std::make_unique<CudaStream>(stream, device, true, nullptr, nullptr);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [external_stream, external_cudnn_handle, external_cublas_handle](const OrtDevice& device) {
      return std::make_unique<CudaStream>(external_stream, device, false, external_cudnn_handle, external_cublas_handle);
    });
}

}