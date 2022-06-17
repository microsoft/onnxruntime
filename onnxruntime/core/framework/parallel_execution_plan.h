// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
//#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/basic_types.h"
#include <unordered_map>

namespace onnxruntime {

class SessionState;
struct ReleasePlan;
struct AllocPlanPerValue;
struct ParallelExecutionPlanImpl; 
class ISequentialPlannerContext;

// Specify how many logic streams for each provider type
using ProviderStreamMap = std::unordered_map<std::string, int>;
// Each set contains ops which should be grouped in an independent logic stream
using OpStreamMap = std::vector<std::vector<std::string>>;

class ParallelExecutionPlan : public IExecutor {
 public:
  ParallelExecutionPlan(const SessionState& session_state);

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;
  std::unique_ptr<ParallelExecutionPlanImpl> impl_;
};

}  // namespace onnxruntime