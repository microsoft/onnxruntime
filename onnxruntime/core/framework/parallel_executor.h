// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <condition_variable>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace onnxruntime {

class ExecutionFrame;

class ParallelExecutor : public IExecutor {
 public:
  ParallelExecutor(const bool& terminate_flag = false) : terminate_flag_{terminate_flag} {}
  ParallelExecutor(const SessionState& session_state, const bool& terminate_flag = false);

  common::Status Execute(const SessionState& session_state,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>& fetches,
                         const logging::Logger& logger) override;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelExecutor);

  void RunNodeAsync(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);
  void RunNodeAsyncInternal(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  void EnqueueNode(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  Status FetchOutput(const MLValueNameIdxMap& name_idx_map,
                     ExecutionFrame& frame,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>& fetches,
                     const logging::Logger& logger);

  void FinishNodeRun() {
    if (--out_standings_ == 0) {
      //std::cout << "all out standing nodes are completed." << std::endl;
      complete_cv_.notify_all();
    }
  }

  std::unique_ptr<ExecutionFrame> root_frame_;
  std::vector<size_t> node_refs_;
  std::mutex ref_mutex_;
  std::atomic<int> out_standings_;
  std::mutex complete_mutex_;
  std::condition_variable complete_cv_;

  const bool& terminate_flag_;
};
}  // namespace onnxruntime
