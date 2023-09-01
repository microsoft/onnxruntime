// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_steps.h"
#include "core/framework/sequential_executor.h"
namespace onnxruntime {
BarrierStep::BarrierStep(size_t id, NodeIndex node_index) : SequentialExecutionPlan::ExecutionStep(node_index),
                                                            barrier_id_{id} {}

Status BarrierStep::Execute(StreamExecutionContext& ctx,
                            size_t /*stream_idx*/,
                            SessionScope& /*session_scope*/,
                            const bool& /*terminate_flag*/,
                            bool& continue_flag) {
  continue_flag = ctx.DecCountDownBarrier(barrier_id_);
  return Status::OK();
}

std::string BarrierStep::ToString() const {
  return ::onnxruntime::MakeString("Set a barrier with id: ",
                                   barrier_id_, ", count: ", 2, ".");
}

WaitOnEPStep::WaitOnEPStep(WaitNotificationFn handle,
                           NotificationIndex idx, NodeIndex node_index) : SequentialExecutionPlan::ExecutionStep(node_index),
                                                                          wait_handle_(handle),
                                                                          notification_idx_(idx) {}

Status WaitOnEPStep::Execute(StreamExecutionContext& ctx,
                             size_t stream_idx,
                             SessionScope& /*session_scope*/,
                             const bool& /*terminate_flag*/,
                             bool& continue_flag) {
  ORT_ENFORCE(wait_handle_, "WaitOnEPStep.wait_handle is null");
  wait_handle_(*ctx.GetDeviceStream(stream_idx), *ctx.GetNotification(notification_idx_));
  // update streams clock status
  if (ctx.GetDeviceStream(stream_idx)) {
    ctx.GetDeviceStream(stream_idx)->UpdateStreamClock(ctx.GetNotification(notification_idx_)->GetStreamSyncTable());
  }
  LOGS(ctx.GetLogger(), VERBOSE) << "stream " << stream_idx << " wait on Notification with id: " << notification_idx_;
  continue_flag = true;
  return Status::OK();
}

std::string WaitOnEPStep::ToString() const {
  return ::onnxruntime::MakeString("WaitOnEPStep: wait on notification with id: ",
                                   notification_idx_, ". ");
}

LaunchKernelStep::LaunchKernelStep(NodeIndex index) : SequentialExecutionPlan::ExecutionStep(index) {}

Status LaunchKernelStep::Execute(StreamExecutionContext& ctx,
                                 size_t stream_idx,
                                 SessionScope& session_scope,
                                 const bool& terminate_flag,
                                 bool& continue_flag) {
#ifdef ENABLE_TRAINING
  // legacy code required by ORTTrainer. Should be removed when ORTTrainer is removed
  auto* node_to_execute = ctx.GetNodeToExecute();
  if (node_to_execute && node_to_execute->count(node_index_) == 0) {
    continue_flag = true;
    return Status::OK();
  }
#endif
  onnxruntime::Status status = ExecuteKernel(ctx, node_index_, stream_idx, terminate_flag, session_scope);
  continue_flag = status.IsOK();
  return status;
}

std::string LaunchKernelStep::ToString() const {
  return ::onnxruntime::MakeString("Launch kernel with node id: ", node_index_, ". ");
}

ActivateNotificationStep::ActivateNotificationStep(
    NotificationIndex notification_index, NodeIndex node_index) : SequentialExecutionPlan::ExecutionStep(node_index),
                                                                  notification_idx_(notification_index) {}

Status ActivateNotificationStep::Execute(StreamExecutionContext& ctx,
                                         size_t stream_idx,
                                         SessionScope& /*session_scope*/,
                                         const bool& /*terminate_flag*/,
                                         bool& continue_flag) {
  if (ctx.GetNotification(notification_idx_)) {
    ctx.GetNotification(notification_idx_)->ActivateAndUpdate();
  }
  LOGS(ctx.GetLogger(), VERBOSE) << "stream " << stream_idx
                                 << " activate notification with index " << notification_idx_;
  continue_flag = true;
  return Status::OK();
}

std::string ActivateNotificationStep::ToString() const {
  return ::onnxruntime::MakeString("ActivateNotificationStep: activate notification with id: ",
                                   notification_idx_, ". ");
}

TriggerDownstreamStep::TriggerDownstreamStep(size_t trigger_point_index, NodeIndex node_index) : SequentialExecutionPlan::ExecutionStep(node_index),
                                                                                                 trigger_point_index_(trigger_point_index) {}

Status TriggerDownstreamStep::Execute(StreamExecutionContext& ctx,
                                      size_t /*stream_idx*/,
                                      SessionScope& session_scope,
                                      const bool& terminate_flag,
                                      bool& continue_flag) {
  ScheduleDownstream(ctx, trigger_point_index_, ctx.SingleThreadMode(), terminate_flag, session_scope);
  continue_flag = true;
  return Status::OK();
}

std::string TriggerDownstreamStep::ToString() const {
  return ::onnxruntime::MakeString("TriggerDownstreamStep: trigger downstream of trigger point: ",
                                   trigger_point_index_, ".");
}
}  // namespace onnxruntime
