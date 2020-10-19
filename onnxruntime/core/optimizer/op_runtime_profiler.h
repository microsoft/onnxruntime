// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/framework/iexecutor.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

class OpRuntimeProfiler {
public:
  explicit OpRuntimeProfiler() {};

  common::Status ProfileGraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                              const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                              const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                              const logging::Logger& logger);
  float GetRuntime(const std::string& name);

private:
  common::Status RunOp(const OpKernel* p_op_kernel, OpKernelContextInternal& op_kernel_context,
                       const Node& node, const logging::Logger& logger);

  std::map<std::string, float> op_runtimes_;

};

}  // namespace onnxruntime
