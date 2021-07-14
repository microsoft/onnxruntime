// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_fence.h"

#include "core/graph/constants.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/gpu_data_transfer.h"

namespace onnxruntime {

ROCMFence::ROCMFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  // NOTE: hipEventBlockingSync may leads to longer wait time because of thread yield/switching in kernel
  // if lower CPU usage is more important than latency, we should use this flag to avoid spin-loop in WaitOnCPU
  int event_flags = /*hipEventBlockingSync |*/ hipEventDisableTiming;
  HIP_CALL_THROW(hipEventCreateWithFlags(&read_event_, event_flags));
  HIP_CALL_THROW(hipEventCreateWithFlags(&write_event_, event_flags));
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
  hipError_t status;
  status = hipEventQuery(read_event_);
  if (status == hipErrorNotReady) {
      // ignore and clear the error if not ready
      hipGetLastError();
      return false;
  } else if (status != hipSuccess) {
      RocmCall<hipError_t, true>(status, "hipEventQuery(read_event_)", "HIP", hipSuccess);
  }
  status = hipEventQuery(write_event_);
  if (status == hipErrorNotReady) {
      // ignore and clear the error if not ready
      hipGetLastError();
      return false;
  } else if (status != hipSuccess) {
      RocmCall<hipError_t, true>(status, "hipEventQuery(write_event_)", "HIP", hipSuccess);
  }
  return true;
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
