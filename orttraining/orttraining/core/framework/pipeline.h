// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gsl/gsl"
#include "orttraining/core/framework/distributed_run_context.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {
namespace training {
namespace pipeline {

// Action in Slot.
// One slot can contain multiple actions and these actions can
// be computed in parallel.
struct PipelineTask {
  // Types of Action.
  // Those types are things we can do in a time slot.
  enum class Type { Empty,
                    Compute,
                    Send,
                    Recv };

  // Passes of Action.
  // A pass is a meaningful sub-graph in the training graph.
  enum class Pass { Unknown,
                    Forward,
                    Backward };

  bool IsForward() const { return pass == Pass::Forward; }
  bool IsBackward() const { return pass == Pass::Backward; }
  bool IsCompute() const { return type == Type::Compute; }
  bool IsCommute() const { return type == Type::Send || type == Type::Recv; }
  bool IsSendTo(const int dst_rank) const {
    if (type != Type::Send) {
      return false;
    }
    return peer_rank == dst_rank;
  }

  bool IsRecvFrom(const int src_rank) const {
    if (type != Type::Recv) {
      return false;
    }
    return peer_rank == src_rank;
  }

  friend std::ostream& operator<<(std::ostream& stream, const PipelineTask& slot);

  // Batch ID this action belongs to.
  // For the first batch's forward pass (Compute), this value is 0.
  // A negative value means undefined.
  int batch;
  Type type;
  Pass pass;

  // The last upstream action' time in compute-only schedule.
  int upstream_time{-1};
  // The last upstream action' stage in compute-only schedule.
  int upstream_stage{-1};

  // The rank where this Action is executed on.
  int this_rank{-1};
  // The remote rank where this Action interact with.
  int peer_rank{-1};

  // This action's scheduled time in compute-commute schedule.
  int full_table_time{-1};
  // This action's scheduled stage in compute-commute schedule.
  int full_table_stage{-1};
};

class PipelineSlot {
 public:
  void AddSend(const int batch_id, const PipelineTask::Pass pass, const int upstream_time = -1, const int upstream_stage = -1, const int this_rank = -1, const int peer_rank = -1);
  void AddRecv(const int batch_id, const PipelineTask::Pass pass, const int upstream_time = -1, const int upstream_stage = -1, const int this_rank = -1, const int peer_rank = -1);
  void AddCompute(const int batch_id, const PipelineTask::Pass pass, const int upstream_time = -1, const int upstream_stage = -1);

  bool IsEmpty() const { return tasks_.empty(); };
  size_t NumActions() const { return tasks_.size(); }

  bool HasCompute() const {
    return std::any_of(
        tasks_.begin(), tasks_.end(), [&](const PipelineTask& task) {
          return task.IsCompute();
        });
  }

  bool HasCommute() const {
    return std::any_of(
        tasks_.begin(), tasks_.end(), [&](const PipelineTask& task) {
          return task.IsCommute();
        });
  }

  bool HasRendTo(const int stage) const {
    return std::any_of(
        tasks_.begin(), tasks_.end(), [&](const PipelineTask& task) {
          return task.IsSendTo(stage);
        });
  }

  bool HasRecvFrom(const int stage) const {
    return std::any_of(
        tasks_.begin(), tasks_.end(), [&](const PipelineTask& task) {
          return task.IsRecvFrom(stage);
        });
  }

  PipelineTask& operator[](int index);
  const PipelineTask& operator[](int index) const;
  PipelineTask& GetFrontAction();
  const PipelineTask& GetFrontAction() const;

  // Print this structure following a fixed-length format.
  // It assumes there are at most 2 actions per slot.
  friend std::ostream& operator<<(std::ostream& stream, const PipelineSlot& slot);
  void SetWaitedEvent(const std::vector<int> event);
  std::vector<int> GetWaitedEvent() const;
  void SetRecordedEvent(const std::vector<int> event);
  std::vector<int> GetRecordedEvent() const;

  std::vector<PipelineTask> GetTasks() { return tasks_; }

  int Size() const { return static_cast<int>(tasks_.size()); }

 private:
  // Actions which can be executed in parallel in this time slot.
  std::vector<PipelineTask> tasks_;
  // Events waited by this slot.
  std::vector<int> waited_events_;
  // Events recorded by this slot.
  std::vector<int> recorded_events_;
};

class PipelineScheduler {
 public:
  PipelineScheduler();
  PipelineScheduler(const int num_batches, const int num_stages, const std::vector<int>& stage_id_to_rank_id_map);

  // Number of time steps.
  size_t GetScheduleSize() const { return compute_commute_table_.size(); }
  // Number of stages.
  size_t GetStageSize() const { return num_stages_; }
  std::vector<PipelineSlot> GetSchedule(const int stage_id) const {
    std::vector<PipelineSlot> commute_slots;
    for (int t = 0; static_cast<size_t>(t) < GetScheduleSize(); ++t) {
      auto& slot = compute_commute_table_.at(t).at(stage_id);
      if (!slot.HasCommute()) {
        continue;
      }
      commute_slots.push_back(slot);
    }
    return commute_slots;
  }

  // APIs to get events for the following pattern.
  //   Wait -> Recv -> Record -> Wait -> Compute -> Record -> Wait -> Send -> Record.
  // If no event exists, -1 may be returned.
  //
  // Forward Recv.
  int GetForwardRecvWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardRecvRecordedEvent(const int batch_id, const int stage_id) const;
  // Forward Compute.
  int GetForwardComputeWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardComputeRecordedEvent(const int batch_id, const int stage_id) const;
  // Forward Send.
  int GetForwardSendWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardSendRecordedEvent(const int batch_id, const int stage_id) const;
  // Backward Recv.
  int GetBackwardRecvWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardRecvRecordedEvent(const int batch_id, const int stage_id) const;
  // Backward Compute.
  int GetBackwardComputeWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardComputeRecordedEvent(const int batch_id, const int stage_id) const;
  // Backward Send.
  int GetBackwardSendWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardSendRecordedEvent(const int batch_id, const int stage_id) const;
  // Visualization of this object.
  friend std::ostream& operator<<(std::ostream& stream, PipelineScheduler const& schedule);

 private:
  // Return time indexes of a given batch.
  // i-th returned element is the time of batch_id's forward at stage i.
  // previous_forward_time[s] is the time that the last forward happens on stage s.
  std::vector<int> FindForwardComputeTime(const std::vector<int> previous_forward_time) const;
  // Return time indexes of a given batch.
  // i-th returned element is the time of batch_id's backward at stage i.
  // forward_time[s] is the forward time for the given batch on stage s.
  std::vector<int> FindBackwardComputeTime(const std::vector<int> forward_time) const;
  void CreateComputeSchedule();
  void InsertEvents(std::vector<std::vector<PipelineSlot>>& schedule, const size_t num_events_per_slot, const std::vector<int> initial_events);
  void CreateFullSchedule();
  void MapStageIdToMpiRank();
  int FindSendRecvTime(const int upstream_compute_time, const int upstream_stage, const int stage) const;
  void InsertForwardCompute(const int batch_id, const std::vector<int> forward_time);
  void InsertBackwardCompute(const int batch_id, const std::vector<int> forward_time, const std::vector<int> backward_time);

  // Given an action, this function finds its home slot and returns events of that slot.
  std::vector<int> TryGetEvent(const bool is_waited_event, const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type, bool& is_found) const;
  // Wrapper over TryGetEvent. It returns -1 when the specified action is not found.
  int GetEventOrDefault(const bool is_waited_event, const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type) const;

  // Number of pipeline stages.
  int num_stages_;
  // Number of micro-batches.
  int num_batches_;
  // Compute-only pipeline schedule as a 2-D table. table_[i][j] is the computation happening in
  // the i-th time slot at the j-th stage. For example, PipeDream schedule may have
  //   1. table_[0][0].batch_id is 0 and table_[0][0].type is Forward.
  //   2. table_[0][1].type is Empty, which means no computation.
  //   3. table_[1][0].batch_id is 1 and table_[1][0].type is Forward.
  std::vector<std::vector<PipelineSlot>> compute_table_;
  // Compute + commute pipeline schedule. Its format is the same as compute-only schedule.
  std::vector<std::vector<PipelineSlot>> compute_commute_table_;
  // Number of active batches per time slot. compute_batch_count_[i] is the number of active
  // compute batches at the i-th time slot.
  std::vector<int> compute_batch_count_;
  std::vector<int> commute_batch_count_;
  // stage_id_to_rank_map[i] is the process' MPI rank to execute stage i inside
  // the current process' pipeline parallel group.
  //
  // Different pipeline parallel groups may have different mapping. For example,
  // if we have 2-stage pipeline parallel with 2-fold data parallel, the 1st pipeline parallel
  // group contains MPI ranks [0, 1] and the 2nd one contains ranks [2, 3].
  // In the 1st pipeline parallel group, stage 0/1 runs on rank 0/1 so stage_id_to_rank_map=[0, 1].
  // For the 2nd group, stage 0/1 runs on rank 2/3 so stage_id_to_rank_map=[2, 3].
  std::vector<int> stage_id_to_rank_id_map_;
};

struct PipelineWorkerState {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;
  std::vector<std::string> fetch_names;
  std::vector<MLValue> fetches;
  std::exception_ptr execution_exception{nullptr};
};

struct PipelineWorkerPool {
  PipelineWorkerPool() = default;
  PipelineWorkerPool(size_t num_workers) : workers(num_workers), worker_states(num_workers){};
  void Join(size_t worker_id);
  void JoinAll();

  std::vector<std::thread> workers;
  std::vector<PipelineWorkerState> worker_states;
};

// Structure to store special tensors' names for pipeline parallel.
struct PipelineTensorNames {
  // Event ops' inputs and outputs related to forward Recv.
  std::string forward_recv_waited_event_name;
  std::string forward_recv_wait_output_name;
  std::string forward_recv_recorded_event_name;
  std::string forward_recv_record_output_name;
  // Event ops' inputs and outputs related to forward Send.
  std::string forward_send_waited_event_name;
  std::string forward_send_wait_output_name;
  std::string forward_send_recorded_event_name;
  std::string forward_send_record_output_name;
  // Event ops' inputs and outputs related to backward Recv.
  std::string backward_recv_waited_event_name;
  std::string backward_recv_wait_output_name;
  std::string backward_recv_recorded_event_name;
  std::string backward_recv_record_output_name;
  // Event ops' inputs and outputs related to backward Send.
  std::string backward_send_waited_event_name;
  std::string backward_send_wait_output_name;
  std::string backward_send_recorded_event_name;
  std::string backward_send_record_output_name;
  // Event ops' inputs and outputs related to forward Compute.
  std::string forward_compute_waited_event_name;
  std::string forward_compute_wait_output_name;
  std::string forward_compute_recorded_event_name;
  std::string forward_compute_record_output_name;
  // Event ops' inputs and outputs related to backward Compute.
  std::string backward_compute_waited_event_name;
  std::string backward_compute_wait_output_name;
  std::string backward_compute_recorded_event_name;
  std::string backward_compute_record_output_name;

  void ForEachEventName(std::function<void(std::string)> fun) {
    fun(forward_recv_waited_event_name);
    fun(forward_recv_recorded_event_name);
    fun(forward_send_waited_event_name);
    fun(forward_send_recorded_event_name);
    fun(backward_recv_waited_event_name);
    fun(backward_recv_recorded_event_name);
    fun(backward_send_waited_event_name);
    fun(backward_send_recorded_event_name);
    fun(forward_compute_waited_event_name);
    fun(forward_compute_recorded_event_name);
    fun(backward_compute_waited_event_name);
    fun(backward_compute_recorded_event_name);
  }

  void ForEachOutputName(std::function<void(std::string)> fun) {
    fun(forward_recv_wait_output_name);
    fun(forward_recv_record_output_name);
    fun(forward_send_wait_output_name);
    fun(forward_send_record_output_name);
    fun(backward_recv_wait_output_name);
    fun(backward_recv_record_output_name);
    fun(backward_send_wait_output_name);
    fun(backward_send_record_output_name);
    fun(forward_compute_wait_output_name);
    fun(forward_compute_record_output_name);
    fun(backward_compute_wait_output_name);
    fun(backward_compute_record_output_name);
  }
};

struct PipelineContext {
  // Number of pipeline stages.
  int num_pipeline_stages;
  // Id of stage handled by this process. Currently, it matches the MPI's rank.
  int pipeline_stage_id;
  // The number of micro-batches per pipeline round.
  // Only the last step among num_gradient_accumulation_steps steps may call
  // optimizer to update the model.
  int num_pipeline_micro_batches;
  // Names of scheduling event in graph's input list and
  // names of event ops' outputs. If an event name is an
  // empty string, it means no event should be waited or recorded.
  PipelineTensorNames pipeline_tensor_names;
  // Allowed feed names.
  // It stands for inputs of a graph partition at this stage.
  std::vector<std::string> feed_names;
  // Allowed fetch names.
  // Values can be fetched at this pipeline stage.
  std::vector<std::string> fetch_names;

  // When running training session with multiple micro-batches, only the last micro-batch run
  // should execute the optimizer nodes and update the model. All non-last micro-batches should
  // only execute until gradient accumulation step.
  std::vector<std::string> accumulation_step_fetches;

  // Outputs of the graph before graph partition.
  std::vector<std::string> expected_output_names;

  // Input and output names of sliced tensors in the original graph.
  std::vector<std::string> sliced_tensor_names;

  // sliced_axes["name"] is the axis to slice along for the tensor called "name".
  std::unordered_map<std::string, int> sliced_axes;

  // sliced_schema["name"] is the shape of sliced version of tensor "name".
  // It's the shape when running micro-batches with pipeline parallel.
  std::unordered_map<std::string, std::vector<int>> sliced_schema;
};

}  // namespace pipeline
}  // namespace training
}  // namespace onnxruntime