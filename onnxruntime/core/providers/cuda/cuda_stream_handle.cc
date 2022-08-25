#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/common/spin_pause.h"

namespace onnxruntime {

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
    
    //CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaEventSynchronize(event_));
  }

  cudaEvent_t event_;
};

CudaStream::CudaStream(cudaStream_t stream, const OrtDevice& device, bool own_flag) : Stream(stream, device), own_stream_(own_flag) {
  if (own_stream_) {
    CUBLAS_CONFIG_CALL(cublasCreate(&cublas_handle_));
    CUBLAS_CONFIG_CALL(cublasSetStream(cublas_handle_, stream));
    CUDNN_CONFIG_CALL(cudnnCreate(&cudnn_handle_));
    CUDNN_CONFIG_CALL(cudnnSetStream(cudnn_handle_, stream));
  }
}

CudaStream::~CudaStream() {
  if (cublas_handle_) {
    CUBLAS_CONFIG_CALL(cublasDestroy(cublas_handle_));
  }
  if (cudnn_handle_) {
    CUDNN_CONFIG_CALL(cudnnDestroy(cudnn_handle_));
  }
  if (handle && own_stream_) {
    CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(handle)));
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

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry, const OrtDevice::DeviceType device_type, cudaStream_t external_stream, bool use_existing_stream) {
  // wait cuda notification on cuda ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitCudaNotificationOnDevice);
  // wait cuda notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitCudaNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [](const OrtDevice& device) {
      cudaStream_t stream = nullptr;
      //CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CALL_THROW(cudaStreamCreate(&stream));
      return std::make_unique<CudaStream>(stream, device, true);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [external_stream](const OrtDevice& device) {
      return std::make_unique<CudaStream>(external_stream, device, false);
    });
}

}