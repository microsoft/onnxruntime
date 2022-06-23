// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/cuda_fence.h"

namespace onnxruntime {

CUDAFence::CUDAFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  // NOTE: cudaEventBlockingSync may leads to longer wait time because of thread yield/switching in kernel
  // if lower CPU usage is more important than latency, we should use this flag to avoid spin-loop in WaitOnCPU
  int event_flags = /*cudaEventBlockingSync |*/ cudaEventDisableTiming;
  CUDA_CALL_THROW(cudaEventCreate(&read_event_, event_flags));
  CUDA_CALL_THROW(cudaEventCreate(&write_event_, event_flags));
}

CUDAFence::~CUDAFence() {
  CUDA_CALL_THROW(cudaEventDestroy(read_event_));
  CUDA_CALL_THROW(cudaEventDestroy(write_event_));
}

void CUDAFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  
}

void CUDAFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  
}

bool CUDAFence::CanRelease() {
  return cudaEventQuery(read_event_) == cudaSuccess &&
         cudaEventQuery(write_event_) == cudaSuccess;
}

void CUDAFence::AfterUsedAsInput(int queue_id) {
  
}

void CUDAFence::AfterUsedAsOutput(int queue_id) {
  
}

}  // namespace onnxruntime
