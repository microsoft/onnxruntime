// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/npu_data_transfer.h"
#include "core/providers/cann/cann_fence.h"

namespace onnxruntime {

CANNFence::CANNFence(const NPUDataTransfer* data_transfer) : data_transfer_(data_transfer) {
  CANN_CALL_THROW(aclrtCreateEvent(&read_event_));
  CANN_CALL_THROW(aclrtCreateEvent(&write_event_));
}

CANNFence::~CANNFence() {
  CANN_CALL_THROW(aclrtDestroyEvent(read_event_));
  CANN_CALL_THROW(aclrtDestroyEvent(write_event_));
}

void CANNFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int async_queue_id) {
  if (provider_type == onnxruntime::kCannExecutionProvider) {
    CANN_CALL_THROW(aclrtStreamWaitEvent(data_transfer_->GetStream(async_queue_id), write_event_));
    CANN_CALL_THROW(aclrtResetEvent(write_event_, data_transfer_->GetStream(async_queue_id)));
  } else {
    CANN_CALL_THROW(aclrtSynchronizeEvent(write_event_));
  }
}

void CANNFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  if (provider_type == onnxruntime::kCannExecutionProvider) {
    aclrtStream stream = data_transfer_->GetStream(queue_id);
    CANN_CALL_THROW(aclrtStreamWaitEvent(stream, read_event_));
    CANN_CALL_THROW(aclrtResetEvent(read_event_, stream));
    CANN_CALL_THROW(aclrtStreamWaitEvent(stream, write_event_));
    CANN_CALL_THROW(aclrtResetEvent(write_event_, stream));
  } else {
    CANN_CALL_THROW(aclrtSynchronizeEvent(read_event_));
    CANN_CALL_THROW(aclrtSynchronizeEvent(write_event_));
  }
}

bool CANNFence::CanRelease() {
  aclrtEventRecordedStatus read_status;
  aclrtEventRecordedStatus write_status;

  return aclrtQueryEventStatus(read_event_, &read_status) == ACL_SUCCESS &&
         aclrtQueryEventStatus(write_event_, &write_status) == ACL_SUCCESS &&
         read_status == ACL_EVENT_RECORDED_STATUS_COMPLETE &&
         write_status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

void CANNFence::AfterUsedAsInput(int queue_id) {
  aclrtStream stream = data_transfer_->GetStream(queue_id);
  CANN_CALL_THROW(aclrtRecordEvent(read_event_, stream));
}

void CANNFence::AfterUsedAsOutput(int queue_id) {
  aclrtStream stream = data_transfer_->GetStream(queue_id);
  CANN_CALL_THROW(aclrtRecordEvent(write_event_, stream));
}

}  // namespace onnxruntime
