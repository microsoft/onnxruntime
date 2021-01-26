// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_fence.h"

#include "core/graph/constants.h"
#include "core/providers/rocm/gpu_data_transfer.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {

ROCMFence::ROCMFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  HIP_CALL_THROW(hipEventCreate(&read_event_));
  HIP_CALL_THROW(hipEventCreate(&write_event_));
}

ROCMFence::~ROCMFence() {
  HIP_CALL_THROW(hipEventDestroy(read_event_));
  HIP_CALL_THROW(hipEventDestroy(write_event_));
}

void ROCMFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  if (provider_type == onnxruntime::kRocmExecutionProvider) {
    // sync in GPU, the call is non-blocking on CPU
    HIP_CALL_THROW(hipStreamWaitEvent(data_transfer_->GetStream(async_queue_id), write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    HIP_CALL_THROW(hipEventSynchronize(write_event_));
  }
}

void ROCMFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  if (provider_type == onnxruntime::kRocmExecutionProvider) {
    // sync in GPU, the call is non-blocking on CPU
    hipStream_t stream = data_transfer_->GetStream(queue_id);
    HIP_CALL_THROW(hipStreamWaitEvent(stream, read_event_, 0));
    HIP_CALL_THROW(hipStreamWaitEvent(stream, write_event_, 0));
  } else {
    // sync on CPU for all other providers, this is blocking
    HIP_CALL_THROW(hipEventSynchronize(read_event_));
    HIP_CALL_THROW(hipEventSynchronize(write_event_));
  }
}

bool ROCMFence::CanRelease() {
  return hipEventQuery(read_event_) == hipSuccess &&
         hipEventQuery(write_event_) == hipSuccess;
}

void ROCMFence::AfterUsedAsInput(int queue_id) {
  // update read fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  HIP_CALL_THROW(hipEventRecord(read_event_, stream));
}

void ROCMFence::AfterUsedAsOutput(int queue_id) {
  // update write fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  HIP_CALL_THROW(hipEventRecord(write_event_, stream));
}

}  // namespace onnxruntime
