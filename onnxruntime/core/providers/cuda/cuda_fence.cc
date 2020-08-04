// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_common.h"
#include "cuda_fence.h"

namespace onnxruntime {

CUDAFence::CUDAFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  // NOTE: cudaEventBlockingSync may leads to longer wait time because of thread yield/switching in kernel
  // if lower CPU usage is more important than latency, we should use this flag to avoid spin-loop in WaitOnCPU
  int event_flags = /*cudaEventBlockingSync |*/ cudaEventDisableTiming;
  for (int i = 0; i < (int)kTotalCudaStreams; ++i)
    CUDA_CALL_THROW(cudaEventCreate(&read_event_[i], event_flags));
  CUDA_CALL_THROW(cudaEventCreate(&write_event_, event_flags));
}

CUDAFence::~CUDAFence() {
  for (int i = 0; i < (int)kTotalCudaStreams; ++i)
    CUDA_CALL_THROW(cudaEventDestroy(read_event_[i]));
  CUDA_CALL_THROW(cudaEventDestroy(write_event_));
}

void CUDAFence::BeforeUsingAsInput(bool sync_cpu, int queue_id) {
  if (!sync_cpu) {
    // sync in GPU, the call is non-blocking on CPU
    CUDA_CALL_THROW(cudaStreamWaitEvent(data_transfer_->GetStream(queue_id), write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    CUDA_CALL_THROW(cudaEventSynchronize(write_event_));
  }
}

void CUDAFence::BeforeUsingAsOutput(bool sync_cpu, int queue_id) {
  if (!sync_cpu) {
    // sync in GPU, the call is non-blocking on CPU
    cudaStream_t stream = data_transfer_->GetStream(queue_id);
    for (int i = 0; i < (int)kTotalCudaStreams; ++i)
      CUDA_CALL_THROW(cudaStreamWaitEvent(stream, read_event_[i], 0));
    CUDA_CALL_THROW(cudaStreamWaitEvent(stream, write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    for (int i = 0; i < (int)kTotalCudaStreams; ++i)
      CUDA_CALL_THROW(cudaEventSynchronize(read_event_[i]));
    CUDA_CALL_THROW(cudaEventSynchronize(write_event_));
  }
}

bool CUDAFence::CanRelease() {
  bool read_done = true;
  for (int i = 0; i < (int)kTotalCudaStreams; ++i)
    read_done = read_done && (cudaEventQuery(read_event_[i]) == cudaSuccess);
  return read_done && cudaEventQuery(write_event_) == cudaSuccess;
}

void CUDAFence::AfterUsedAsInput(int queue_id) {
  // update read fence
  cudaStream_t stream = data_transfer_->GetStream(queue_id);
  CUDA_CALL_THROW(cudaEventRecord(read_event_[queue_id], stream));
}

void CUDAFence::AfterUsedAsOutput(int queue_id) {
  // update write fence
  cudaStream_t stream = data_transfer_->GetStream(queue_id);
  CUDA_CALL_THROW(cudaEventRecord(write_event_, stream));
}

}  // namespace onnxruntime
