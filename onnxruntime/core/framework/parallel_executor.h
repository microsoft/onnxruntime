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
  ParallelExecutor() = default;
  ParallelExecutor(const SessionState& session_state);

  common::Status Execute(const SessionState& session_state,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>& fetches,
                         const logging::Logger& logger) override;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelExecutor);

  void RunNodeAsync(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  void EnqueueNode(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  Status FetchOutput(const MLValueNameIdxMap& name_idx_map,
                     ExecutionFrame& frame,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>& fetches,
                     const logging::Logger& logger);

  std::unique_ptr<ExecutionFrame> root_frame_;
  std::vector<size_t> node_refs_;
  std::mutex ref_mutex_;
  std::atomic<int> out_standings_;
  std::mutex complete_mutex_;
  std::condition_variable complete_cv_;
};
}  // namespace onnxruntime
