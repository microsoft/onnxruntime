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
    //activate the notification.
    ready_.store(true); 
  }

  bool wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.provider->Type() == kCudaExecutionProvider);
    // wait for the notification to be activated
    //while (!ready_.load()) {
    //  onnxruntime::concurrency::SpinPause();
    //}
    if (!ready_.load()) {
      return false;
    }
    // launch a wait command to the cuda stream
    CUDA_CALL_THROW(cudaStreamWaitEvent(static_cast<cudaStream_t>(device_stream.handle), 
                                        event_));
    return true;
  };

  bool wait_on_host() {
    // wait for the notification to be activated
    //while (!ready_.load()) {
    //  onnxruntime::concurrency::SpinPause();
    //}
    if (!ready_.load()) {
      return false;
    }
    //CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaEventSynchronize(event_));
    return true;
  }

  std::atomic_bool ready_{};
  cudaEvent_t event_;
};

CudaStream::CudaStream(cudaStream_t stream, const IExecutionProvider* ep) : Stream(stream, ep) {
}

CudaStream::~CudaStream(){
  if (handle)
    CUDA_CALL(cudaStreamDestroy(static_cast<cudaStream_t>(handle)));
}

std::unique_ptr<synchronize::Notification> CudaStream::CreateNotification(size_t /*num_consumers*/){
  return std::make_unique<CudaNotification>(this);
}

void CudaStream::Flush(){
  CUDA_CALL_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(handle))); 
}


// CPU Stream command handles
bool WaitCudaNotificationOnDevice(Stream& stream, synchronize::Notification& notification) {
  return static_cast<CudaNotification*>(&notification)->wait_on_device(stream);
}

bool WaitCudaNotificationOnHost(Stream& /*stream*/, synchronize::Notification& notification) {
  return static_cast<CudaNotification*>(&notification)->wait_on_host();
}

void ReleaseCUdaNotification(void* handle) {
  delete static_cast<CudaNotification*>(handle);
}

std::unique_ptr<Stream> CreateCudaStream(const IExecutionProvider* provider) {
  ORT_ENFORCE(provider->Type() == kCudaExecutionProvider);
  cudaStream_t stream = nullptr;
  //Todo: should we use cudaStreamNonBlocking flag
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  return std::make_unique<CudaStream>(stream, provider);
}

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry) {
  // wait cuda notification on cuda ep
  stream_handle_registry.RegisterWaitFn(kCudaExecutionProvider, kCudaExecutionProvider, WaitCudaNotificationOnDevice);
  // wait cuda notification on cpu ep
  stream_handle_registry.RegisterWaitFn(kCudaExecutionProvider, kCpuExecutionProvider, WaitCudaNotificationOnHost);

  stream_handle_registry.RegisterCreateStreamFn(kCudaExecutionProvider, CreateCudaStream);
}

}