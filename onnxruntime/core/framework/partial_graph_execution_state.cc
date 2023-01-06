//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#include "core/framework/session_state.h"
#include "core/framework/stream_execution_context.h"
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

  new_region.stream_pc_range.reserve(plan->execution_plan.size());
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

PartialGraphExecutionState::~PartialGraphExecutionState() {
}

DeviceStreamCollection* PartialGraphExecutionState::GetDeviceStreamCollection(const SessionState& session_state) {
  if (device_stream_collection_ == nullptr) {
    device_stream_collection_ = session_state.AcquireDeviceStreamCollection();
    // the life-time of partial graph execution state is in-consistant with session,
    // so we can't make sure it is safe to return the device stream collection to
    // session when deconstruct partial graph execution state.
    // so let's always delete the stream collections.
    // luckily, for ort module, we always running with default stream, so no impact to perf.
  }
  return device_stream_collection_.get();
}

StreamExecutionContext& PartialGraphExecutionState::GetExecutionContext(gsl::span<const int>& feed_mlvalue_idxs, gsl::span<const OrtValue>& feeds,
                                                                        gsl::span<const int>& fetch_mlvalue_idxs, std::vector<OrtValue>& fetches,
                                                                        const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                                                        const SessionState& session_state,
                                                                        const logging::Logger& sess_logger,
                                                                        const DeviceStreamCollection* device_streams) {
  if (execution_context_ == nullptr) {
    auto* execution_plan = session_state.GetExecutionPlan();
    LOGS(sess_logger, INFO) << "Number of streams: " << execution_plan->execution_plan.size();
    int32_t valid_streams = 0;
    for (auto& stream : execution_plan->execution_plan) {
      if (stream && stream->steps_.size() > 0)
        valid_streams++;
    }

    execution_context_ = std::make_unique<StreamExecutionContext>(
        session_state,
        valid_streams,
        execution_plan->notification_owners,
        execution_plan->num_barriers,
        device_streams,
        feed_mlvalue_idxs,
        feeds,
        fetch_mlvalue_idxs,
        fetches,
        fetch_allocators,
        sess_logger,
        // partial executor in training can only be run with single thread
        true);
  } else {
    execution_context_->GetExecutionFrame().UpdateFeeds(feed_mlvalue_idxs, feeds);
    execution_context_->GetExecutionFrame().UpdateFetches(fetch_mlvalue_idxs, fetches, session_state.GetInitializedTensors());
    execution_context_->SetLogger(sess_logger);
  }

  return *execution_context_;
}

}  // namespace onnxruntime

#endif
