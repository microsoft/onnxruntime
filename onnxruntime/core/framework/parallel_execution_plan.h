// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include "core/framework/iexecutor.h"

namespace onnxruntime {

class SessionState;
struct ParallelExecutionPlanImpl;

struct ParallelExecutionPlan : public IExecutor {
  ParallelExecutionPlan(const SessionState& session_state, int num_logic_streams);
  ~ParallelExecutionPlan();
  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;
  std::unique_ptr<ParallelExecutionPlanImpl> impl_;
  //ParallelExecutionPlanImpl* impl_;
};

}  // namespace onnxruntime