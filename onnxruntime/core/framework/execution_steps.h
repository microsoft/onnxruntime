// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/sequential_execution_plan.h"
#include "core/framework/stream_execution_context.h"

namespace onnxruntime {

class SessionScope;

class BarrierStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  BarrierStep(size_t id);

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
  WaitOnEPStep(WaitNotificationFn handle, NotificationIndex idx);

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
  LaunchKernelStep(NodeIndex index);

  Status Execute(StreamExecutionContext& ctx,
                 size_t stream_idx,
                 SessionScope& session_scope,
                 const bool& terminate_flag,
                 bool& continue_flag) override;

  std::string ToString() const override;

  NodeIndex GetNodeIndex() { return node_index_; }

 private:
  NodeIndex node_index_{0};
};

class ActivateNotificationStep : public SequentialExecutionPlan::ExecutionStep {
 public:
  ActivateNotificationStep(NotificationIndex notification_index);

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
  TriggerDownstreamStep(size_t trigger_point_index);
  Status Execute(StreamExecutionContext& ctx,
                 size_t /*stream_idx*/,
                 SessionScope& session_scope,
                 const bool& terminate_flag,
                 bool& continue_flag) override;

  virtual std::string ToString() const override;

 private:
  size_t trigger_point_index_;
};
#ifdef ENABLE_TRAINING
/// <summary>
/// Retrieve Region start and end from Program Counter's start and end.
/// The returned start_region and end_region are the index of the argument steps within start_pc and end_pc
/// which is the topological order index of the nodes.
/// </summary>
void RetrieveRegionBoundaryFromProgramCounter(std::vector<std::unique_ptr<SequentialExecutionPlan::ExecutionStep>>& steps,
  const InlinedHashMap<NodeIndex, size_t>& node_index_2_toposort_index,
  size_t start_pc, size_t end_pc, size_t& start_region, size_t& end_region);
#endif
}  // namespace onnxruntime
