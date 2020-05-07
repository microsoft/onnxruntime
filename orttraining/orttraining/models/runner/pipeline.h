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

struct Slot {
  enum Type { Empty, Forward, Backward };
  
  Slot(); 
  bool IsEmpty() const; 
  bool IsForward() const; 
  bool IsBackward() const; 
  void Show() const; 

  Type type;
  int batch_id;
  std::vector<int> waited_events;
  std::vector<int> recorded_events;
};

class PipelineSchedule {
public:
  PipelineSchedule() = default;
  PipelineSchedule(int num_stages);
  void Add(int batch_id);
  void Add(int batch_id_begin, int batch_id_end);
  int GetForwardWaitedEventId(int stage_id, int batch_id) const;
  int GetForwardRecordedEventId(int stage_id, int batch_id) const;
  int GetBackwardWaitedEventId(int stage_id, int batch_id) const;
  int GetBackwardRecordedEventId(int stage_id, int batch_id) const;
  void Show() const;

private:
  std::vector<int> SearchLastRecordedEvents(int time_id, int stage_id) const;

  // 2-D table of pipeline schedule. table_[i][j] is the computation happening in
  // the i-th time slot at the j-th stage. For example, PipeDream schedule may have
  //   1. table_[0][0].batch_id is 0 and table_[0][0].type is Forward.
  //   2. table_[0][1].type is Empty, which means no computation.
  //   3. table_[1][0].batch_id is 1 and table_[1][0].type is Forward.
  std::vector<std::vector<Slot>> table_;
  // Number of active batches per time slot. batch_count_[i] is the number of active
  // batches at the i-th time slot. 
  std::vector<int> batch_count_;
  // Total number of stages of this pipeline. 
  // It equals to table_.size().
  int num_stages_;
  // Total number of batches scheduled in this pipeline. 
  // It equals to table_[i].size(), for i = 0, ..., num_stages_.
  int num_batches_;
};

struct PipelineWorkerState {
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;
  std::vector<std::string> fetch_names;
  std::vector<MLValue> fetches;
};

struct PipelineWorkerPool {
  PipelineWorkerPool() = default;
  PipelineWorkerPool(size_t num_workers) : workers(num_workers), worker_states(num_workers) {};
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
  // [TODO] Add events for communication stages and computation stages independently.
  // std::string forward_comm_waited_event_name;
  // std::string forward_comm_recorded_event_name;
  // std::string forward_comp_waited_event_name;
  // std::string forward_comp_recorded_event_name;
  // std::string backward_comm_waited_event_name;
  // std::string backward_comm_recorded_event_name;
  // std::string backward_comp_waited_event_name;
  // std::string backward_comp_recorded_event_name;
  std::string forward_waited_event_name;
  std::string forward_recorded_event_name;
  std::string backward_waited_event_name;
  std::string backward_recorded_event_name;

  std::string forward_waited_output_name;
  std::string forward_recorded_output_name;
  std::string backward_waited_output_name;
  std::string backward_recorded_output_name;

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