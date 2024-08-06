// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/sequential_execution_plan.h"
#include "core/framework/stream_execution_context.h"

namespace onnxruntime {

class SessionScope;

class BarrierStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  BarrierStep(size_t id, NodeIndex node_index);

  Status Execute(StreamExecutionContext& ctx,
                 size_t /*stream_idx*/,
                 SessionScope& /*session_scope*/,
                 const bool& /*terminate_flag*/,
                 bool& continue_flag) override;

  std::string ToString() const override;

 private:
  size_t barrier_id_{0};
};

class WaitOnEPStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  WaitOnEPStep(WaitNotificationFn handle, NotificationIndex idx, NodeIndex node_index);

  Status Execute(StreamExecutionContext& ctx,
                 size_t stream_idx,
                 SessionScope& /*session_scope*/,
                 const bool& /*terminate_flag*/,
                 bool& continue_flag) override;

  std::string ToString() const override;

 private:
  WaitNotificationFn wait_handle_;
  NotificationIndex notification_idx_;
};

class LaunchKernelStep : public SequentialExecutionPlan::ExecutionStep {
 public:
#if defined(ORT_MINIMAL_BUILD)
  LaunchKernelStep(NodeIndex index);
#else
  LaunchKernelStep(NodeIndex index, std::string_view node_name);
#endif

  Status Execute(StreamExecutionContext& ctx,
                 size_t stream_idx,
                 SessionScope& session_scope,
                 const bool& terminate_flag,
                 bool& continue_flag) override;

  std::string ToString() const override;

#if !defined(ORT_MINIMAL_BUILD)
 private:
  std::string node_name_;
#endif
};

class ActivateNotificationStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  ActivateNotificationStep(NotificationIndex notification_index, NodeIndex node_index);

  Status Execute(StreamExecutionContext& ctx,
                 size_t stream_idx,
                 SessionScope& /*session_scope*/,
                 const bool& /*terminate_flag*/,
                 bool& continue_flag) override;

  virtual std::string ToString() const override;

 private:
  NotificationIndex notification_idx_;
};

class TriggerDownstreamStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  TriggerDownstreamStep(size_t trigger_point_index, NodeIndex node_index);
  Status Execute(StreamExecutionContext& ctx,
                 size_t /*stream_idx*/,
                 SessionScope& session_scope,
                 const bool& terminate_flag,
                 bool& continue_flag) override;

  virtual std::string ToString() const override;

 private:
  size_t trigger_point_index_;
};
}  // namespace onnxruntime
