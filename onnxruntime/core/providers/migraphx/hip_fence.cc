// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "migraphx_inc.h"
#include "hip_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

HIPFence::HIPFence(const GPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  hipEventCreate(&read_event_);
  hipEventCreate(&write_event_);
}

HIPFence::~HIPFence() {
  hipEventDestroy(read_event_);
  hipEventDestroy(write_event_);
}

void HIPFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  (void)provider_type;
  (void)async_queue_id;
  // sync on CPU for all other providers, this is blocking
  hipEventSynchronize(write_event_);
}

void HIPFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  (void)provider_type;
  (void)queue_id;
  
  // sync on CPU for all other providers, this is blocking
  hipEventSynchronize(read_event_);
  hipEventSynchronize(write_event_);
}

bool HIPFence::CanRelease() {
  return hipEventQuery(read_event_) == hipSuccess &&
         hipEventQuery(write_event_) == hipSuccess;
}

void HIPFence::AfterUsedAsInput(int queue_id) {
  // update read fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  hipEventRecord(read_event_, stream);
}

void HIPFence::AfterUsedAsOutput(int queue_id) {
  // update write fence
  hipStream_t stream = data_transfer_->GetStream(queue_id);
  hipEventRecord(write_event_, stream);
}

}  // namespace onnxruntime
