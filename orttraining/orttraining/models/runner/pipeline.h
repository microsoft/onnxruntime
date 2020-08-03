// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "gsl/gsl"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"
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
    for (auto& task : tasks_) {
      if (task.IsCompute())
        return true;
    }
    return false;
  }

  bool HasRendTo(const int stage) const {
    for (auto& task : tasks_) {
      if (task.IsSendTo(stage)) {
        return true;
      }
    }
    return false;
  }

  bool HasRecvFrom(const int stage) const {
    for (auto& task : tasks_) {
      if (task.IsRecvFrom(stage)) {
        return true;
      }
    }
    return false;
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

 private:
  // Actions which can be executed in parallel in this time slot.
  std::vector<PipelineTask> tasks_;

  // For MPI PipeDream schedule, it's used to support Wait -> Recv -> Wait -> Compute -> Record -> Send -> Record.
  // Since Send, Recv, and Compute are stored in the same slot, each slot contains two waited events and two recorded events.
  //
  // For NCCL PipDream schedule, it's used to support Wait -> Recv -> Record -> Wait -> Compute -> Record -> Wait -> Send -> Record.

  // Events waited by this slot.
  std::vector<int> waited_events_;
  // Events recorded by this slot.
  std::vector<int> recorded_events_;
};

class PipelineScheduler {
 public:
  PipelineScheduler(const int num_batches, const int num_stages);
  size_t GetScheduleSize() const { return compute_commute_table_.size(); }
  size_t GetStageSize() const { return num_stages_; }
  // APIs to get NCCL event for
  // Wait -> Recv -> Record -> Wait -> Compute -> Record -> Wait -> Send -> Record.
  int GetForwardComputeWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardComputeRecordedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardComputeWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardComputeRecordedEvent(const int batch_id, const int stage_id) const;
  int GetForwardSendWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardSendRecordedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardSendWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardSendRecordedEvent(const int batch_id, const int stage_id) const;
  int GetForwardRecvWaitedEvent(const int batch_id, const int stage_id) const;
  int GetForwardRecvRecordedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardRecvWaitedEvent(const int batch_id, const int stage_id) const;
  int GetBackwardRecvRecordedEvent(const int batch_id, const int stage_id) const;
  // APIs to get MPI event event for
  // Wait -> Recv -> Wait -> Compute -> Record -> Send -> Record.
  int GetForwardWaitedEventBeforeRecv(const int batch_id, const int stage_id) const;
  int GetForwardWaitedEventAfterRecv(const int batch_id, const int stage_id) const;
  int GetForwardRecordedEventBeforeSend(const int batch_id, const int stage_id) const;
  int GetForwardRecordedEventAfterSend(const int batch_id, const int stage_id) const;
  int GetBackwardWaitedEventBeforeRecv(const int batch_id, const int stage_id) const;
  int GetBackwardWaitedEventAfterRecv(const int batch_id, const int stage_id) const;
  int GetBackwardRecordedEventBeforeSend(const int batch_id, const int stage_id) const;
  int GetBackwardRecordedEventAfterSend(const int batch_id, const int stage_id) const;
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
  int FindSendRecvTime(const int upstream_compute_time, const int upstream_stage, const int stage) const;
  void InsertForwardCompute(const int batch_id, const std::vector<int> forward_time);
  void InsertBackwardCompute(const int batch_id, const std::vector<int> forward_time, const std::vector<int> backward_time);

  // Given an action, this function finds its home slot and returns events of that slot.
  std::vector<int> TryGetEvent(const bool is_waited_event, const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type, bool& is_found) const;
  // Wrapper over TryGetEvent. It returns -1 when the specified action is not found.
  int GetEventOrDefault(const bool is_waited_event, const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type) const;

  std::vector<int> TryGetComputeEvent(const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type, bool& is_found) const;
  int GetComputeEventOrDefault(const bool is_waited_event, const int batch_id, const int stage_id, const PipelineTask::Pass pass, const PipelineTask::Type type) const;

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
  int num_stages_;
  int num_batches_;
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

struct PipelineContext {
  // Id of stage handled by this process. Currently, it matches the MPI's rank.
  int pipeline_stage_id;
  // The number of batches per pipeline run. Its value is
  // num_gradient_accumulation_steps.
  // Only the last step among num_gradient_accumulation_steps steps may call
  // optimizer to update the model.
  int num_pipeline_batches;

  // Name of scheduling event in graph's input list.
  // If an event name is an empty string, it means no event
  // should be waited or recorded.
  std::string forward_waited_event_name;
  std::string forward_waited_event_after_recv_name;
  std::string forward_recorded_event_before_send_name;
  std::string forward_recorded_event_name;
  std::string backward_waited_event_name;
  std::string backward_waited_event_after_recv_name;
  std::string backward_recorded_event_before_send_name;
  std::string backward_recorded_event_name;

  std::string forward_wait_output_name;
  std::string forward_record_output_name;
  std::string backward_wait_output_name;
  std::string backward_record_output_name;

  // Allowed feed names.
  // It stands for inputs of a graph partition at this stage.
  std::vector<std::string> feed_names;
  // Allowed fetch names.
  // Values can be fetched at this pipeline stage.
  std::vector<std::string> fetch_names;
};

}  // namespace pipeline
}  // namespace training
}  // namespace onnxruntime