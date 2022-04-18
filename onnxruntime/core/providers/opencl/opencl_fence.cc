// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_fence.h"

namespace onnxruntime {

OpenCLFence::OpenCLFence(const OpenCLExecutionProvider& exec)
    : produced_(nullptr), consumed_(nullptr), exec_{&exec} {
}

OpenCLFence::~OpenCLFence() {
  if (produced_ != nullptr) {
    clReleaseEvent(produced_);
  }
  if (consumed_ != nullptr) {
    clReleaseEvent(consumed_);
  }
}

void OpenCLFence::BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) {
  ZoneScopedN("BeforeUsingAsInput");
  // Graph nodes are visited by topological order, it should at least ensure the event is created
  ORT_ENFORCE(produced_ != 0);

  auto cmd_queue = exec_->GetCommandQueue(queue_id);
  if (provider_type == kOpenCLExecutionProvider) {
    ORT_THROW_IF_CL_ERROR(clEnqueueMarkerWithWaitList(cmd_queue, 1, &produced_, nullptr));
  } else {
    ORT_THROW_IF_CL_ERROR(clWaitForEvents(1, &produced_));
  }
}

void OpenCLFence::BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) {
  // This API is weird because it is designed for Tensor inplace reuse. In this case, we need to ensure the previous
  // execution output is produced and consumed. OpenCL EP currently do not have memory pattern, thus, no tensor reuse,
  // so it is not implemented. See https://github.com/microsoft/onnxruntime/pull/879#issuecomment-487670768
  ZoneScopedN("BeforeUsingAsOutput");
  ORT_UNUSED_PARAMETER(provider_type);
  ORT_UNUSED_PARAMETER(queue_id);
}

void OpenCLFence::AfterUsedAsInput(int queue_id) {
  ZoneScopedN("AfterUsedAsInput");
  ORT_THROW_IF_CL_ERROR(clEnqueueMarkerWithWaitList(exec_->GetCommandQueue(queue_id), 0, nullptr, &consumed_));
}

void OpenCLFence::AfterUsedAsOutput(int queue_id) {
  ZoneScopedN("AfterUsedAsOutput");
  ORT_THROW_IF_CL_ERROR(clEnqueueMarkerWithWaitList(exec_->GetCommandQueue(queue_id), 0, nullptr, &produced_));
}

bool OpenCLFence::CanRelease() {
  ZoneScopedN("CanRelease");
  ORT_ENFORCE(produced_ != nullptr && consumed_ != nullptr);

  cl_int consume_status = CL_QUEUED;
  ORT_THROW_IF_CL_ERROR(clGetEventInfo(consumed_, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &consume_status, nullptr));

#ifndef NDEBUG
  // If it has been consumed, then it must has been produced also
  cl_int produce_status = CL_QUEUED;
  ORT_THROW_IF_CL_ERROR(clGetEventInfo(produced_, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &produce_status, nullptr));
  if (consume_status == CL_COMPLETE) {
    ORT_ENFORCE(produce_status == CL_COMPLETE);
  }
#endif

  return consume_status == CL_COMPLETE;
}

}  // namespace onnxruntime
