#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_context.h"
#include "core/framework/execution_frame.h"

namespace onnxruntime {

ProgramRegion& PartialGraphExecutionState::GetProgramRegions(const SessionState& session_state) {
  // check whether we can match an existing region
  auto it = std::find_if(program_regions_.begin(), program_regions_.end(),
      [this](const ProgramRegion& region) {
         return region.node_start == this->GetProgramCounterStart() && region.node_end == this->GetProgramCounterEnd();
      });
  if (it != program_regions_.end()) {
    return *it;
  }
  // calculate the new region
  ProgramRegion new_region;
  new_region.node_start = program_counter_start_;
  new_region.node_end = program_counter_end_;
  auto* plan = session_state.GetExecutionPlan();
  for (auto& stream : plan->execution_plan) {
    size_t cur = 0;
    while (cur < stream->step_node_index.size() &&
           stream->step_node_index[cur] < new_region.node_start) {
      cur++;
    }
    size_t start = cur;
    while (cur < stream->step_node_index.size() &&
           stream->step_node_index[cur] <= new_region.node_end) {
      cur++;
    }
    new_region.stream_pc_range.push_back({start, cur});
  }
  program_regions_.push_back(std::move(new_region));
  return program_regions_.back();
}

ExecutionContext& PartialGraphExecutionState::GetExecutionContext(gsl::span<const int>& feed_mlvalue_idxs, gsl::span<const OrtValue>& feeds,
                                                                  gsl::span<const int>& fetch_mlvalue_idxs, std::vector<OrtValue>& fetches,
    const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
    const SessionState& session_state,
    const logging::Logger& sess_logger,
    const DeviceStreamCollection& device_streams_map,
    const bool& terminate_flag) {
  if (execution_context_ == nullptr) {
    auto* execution_plan = session_state.GetExecutionPlan();
    LOGS(sess_logger, INFO) << "Number of streams: " << execution_plan->execution_plan.size();
    int32_t valid_streams = 0;
    for (auto& stream : execution_plan->execution_plan) {
      if (stream && stream->steps_.size() > 0)
        valid_streams++;
    }

    execution_context_ = std::make_unique<ExecutionContext>(
        session_state,
        valid_streams,
        execution_plan->notification_owners,
        feed_mlvalue_idxs,
        feeds,
        fetch_mlvalue_idxs,
        fetches,
        fetch_allocators,
        execution_plan->num_barriers,
        sess_logger,
        device_streams_map,
        terminate_flag,
        // partial executor in training can only be run with single thread
        true);
  } else {
    execution_context_->GetExecutionFrame()->UpdateFeeds(feed_mlvalue_idxs, feeds);
    execution_context_->GetExecutionFrame()->UpdateFetches(fetch_mlvalue_idxs, fetches, session_state.GetInitializedTensors());
  }

  return *execution_context_;
}

}

#endif
