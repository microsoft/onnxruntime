// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ml_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/session/IOBinding.h"
namespace onnxruntime {
class SequentialExecutor : public IExecutor {
 public:
  SequentialExecutor(size_t program_counter_start, size_t program_counter_end, PartialGraphExecutionState& state,
                     const bool& terminate_flag = false, const bool only_execute_path_to_fetches = false)
      : program_counter_start_{program_counter_start}, program_counter_end_{program_counter_end}, state_{&state}, 
      terminate_flag_{terminate_flag}, only_execute_path_to_fetches_(only_execute_path_to_fetches) {}

  SequentialExecutor(const bool& terminate_flag = false, const bool only_execute_path_to_fetches = false)
      : terminate_flag_{terminate_flag}, only_execute_path_to_fetches_(only_execute_path_to_fetches) {}

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;

  size_t program_counter_start_;
  size_t program_counter_end_;
  PartialGraphExecutionState* state_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SequentialExecutor);
  const bool& terminate_flag_;
  const bool only_execute_path_to_fetches_;
};
}  // namespace onnxruntime
