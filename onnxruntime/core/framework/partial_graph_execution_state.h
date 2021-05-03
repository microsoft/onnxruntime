//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_TRAINING
#include "core/common/common.h"
#include "core/framework/execution_frame.h"

namespace onnxruntime {
struct PartialGraphExecutionState {
 public:
  PartialGraphExecutionState() {
    execution_frame_ = nullptr;
    release_outputs_ = false;
  }

  ~PartialGraphExecutionState() = default;

  void SetProgramCounterStart(size_t start) { program_counter_start_ = start; }
  void SetProgramCounterEnd(size_t end) { program_counter_end_ = end; }

  size_t GetProgramCounterStart() { return program_counter_start_; }
  size_t GetProgramCounterEnd() { return program_counter_end_; }

  void SetReleaseOutputs(bool release) { release_outputs_ = release; }
  bool GetReleaseOutputs() { return release_outputs_; }

  ExecutionFrame& GetExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                                    const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                                    const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                    const SessionState& session_state) {
    if (execution_frame_ == nullptr) {
      execution_frame_ = std::make_unique<ExecutionFrame>(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches,
                                                          fetch_allocators, session_state);
    } else {
      execution_frame_->UpdateFeeds(feed_mlvalue_idxs, feeds);
      execution_frame_->UpdateFetches(fetch_mlvalue_idxs, fetches, session_state.GetInitializedTensors());
    }

    return *execution_frame_;
  }

 private:
  std::unique_ptr<ExecutionFrame> execution_frame_;
  size_t program_counter_start_;
  size_t program_counter_end_;
  bool release_outputs_;
};
}  // namespace onnxruntime
#endif
