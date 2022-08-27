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
                           return region.start_pc == this->GetProgramCounterStart() && region.end_pc == this->GetProgramCounterEnd();
                         });
  if (it != program_regions_.end()) {
    return *it;
  }
  auto* plan = session_state.GetExecutionPlan();
  // calculate the new region
  ProgramRegion new_region;
  new_region.start_pc = program_counter_start_;
  new_region.end_pc = program_counter_end_;

  for (auto& stream : plan->execution_plan) {
    size_t cur = 0;
    while (cur < stream->step_pc.size() &&
           stream->step_pc[cur] < new_region.start_pc) {
      cur++;
    }
    size_t start = cur;
    while (cur < stream->step_pc.size() &&
           stream->step_pc[cur] < new_region.end_pc) {
      cur++;
    }
    new_region.stream_pc_range.push_back({start, cur});
  }
  program_regions_.push_back(std::move(new_region));
  return program_regions_.back();
}

std::shared_ptr<ExecutionFrame> PartialGraphExecutionState::GetExecutionFrame(gsl::span<const int> feed_mlvalue_idxs,
                                                                              gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                                                              gsl::span<const OrtValue> fetches,
                                                                              const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                                              const SessionState& session_state,
                                                                              const std::vector<Stream*>* device_streams) {
  if (execution_frame_ == nullptr) {
    execution_frame_ = std::make_shared<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches,
                                                        fetch_allocators, session_state, device_streams);
  } else {
    execution_frame_->UpdateFeeds(feed_mlvalue_idxs, feeds);
    execution_frame_->UpdateFetches(fetch_mlvalue_idxs, fetches, session_state.GetInitializedTensors());
  }

  return execution_frame_;
}

}  // namespace onnxruntime

#endif
