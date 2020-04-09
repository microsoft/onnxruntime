// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_common.h"
#include "cuda_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

CUDAFence::CUDAFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  // NOTE: cudaEventBlockingSync may leads to longer wait time because of thread yield/switching in kernel
  // if lower CPU usage is more important than latency, we should use this flag to avoid spin-loop in WaitOnCPU
  int event_flags = /*cudaEventBlockingSync |*/ cudaEventDisableTiming;
  CUDA_CALL_THROW(cudaEventCreate(&read_event_, event_flags));
  CUDA_CALL_THROW(cudaEventCreate(&write_event_, event_flags));
  // int cuda_device;
  // CUDA_CALL_THROW(cudaGetDevice(&cuda_device));
  // std::cout << "(create) GPU: " << cuda_device << ", read event: " << &read_event_ << std::endl;;
  // std::cout << "(create) GPU: " << cuda_device << ", write event: " << &read_event_ << std::endl;;
}

CUDAFence::~CUDAFence() {
  CUDA_CALL_THROW(cudaEventDestroy(read_event_));
  CUDA_CALL_THROW(cudaEventDestroy(write_event_));
}

void CUDAFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  if (provider_type == onnxruntime::kCudaExecutionProvider) {
    // sync in GPU, the call is non-blocking on CPU
    CUDA_CALL_THROW(cudaStreamWaitEvent(data_transfer_->GetStream(async_queue_id), write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    CUDA_CALL_THROW(cudaEventSynchronize(write_event_));
  }
}

void CUDAFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  if (provider_type == onnxruntime::kCudaExecutionProvider) {
    // sync in GPU, the call is non-blocking on CPU
    cudaStream_t stream = data_transfer_->GetStream(queue_id);
    CUDA_CALL_THROW(cudaStreamWaitEvent(stream, read_event_, 0));
    CUDA_CALL_THROW(cudaStreamWaitEvent(stream, write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    CUDA_CALL_THROW(cudaEventSynchronize(read_event_));
    CUDA_CALL_THROW(cudaEventSynchronize(write_event_));
  }
}

bool CUDAFence::CanRelease() {
  return cudaEventQuery(read_event_) == cudaSuccess &&
         cudaEventQuery(write_event_) == cudaSuccess;
}

void CUDAFence::AfterUsedAsInput(int queue_id) {
  // update read fence
  cudaStream_t stream = data_transfer_->GetStream(queue_id);
  // cudaDeviceSynchronize();

  // int cuda_device;
  // CUDA_CALL_THROW(cudaGetDevice(&cuda_device));
  // std::cout << "(record) GPU: " << cuda_device << ", stream" << queue_id << ", read event: " << &read_event_ << std::endl;;

  CUDA_CALL_THROW(cudaEventRecord(read_event_, stream));
  // cudaDeviceSynchronize();
}

void CUDAFence::AfterUsedAsOutput(int queue_id) {
  // update write fence
  cudaStream_t stream = data_transfer_->GetStream(queue_id);

  int cuda_device;
  CUDA_CALL_THROW(cudaGetDevice(&cuda_device));
  std::cout << "(record) GPU: " << cuda_device << ", stream" << queue_id << ", write event: " << &read_event_ << std::endl;;

  CUDA_CALL_THROW(cudaEventRecord(write_event_, stream));
}

}  // namespace onnxruntime
