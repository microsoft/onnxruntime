// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/op_runtime_profiler.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/graph/graph_viewer.h"

#include <algorithm>
#include <chrono>

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status OpRuntimeProfiler::ProfileGraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                                     const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                     const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                     const logging::Logger& logger) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& feed_mlvalue_idxs = feeds_fetches_info.feeds_mlvalue_idxs;
  const auto& fetch_mlvalue_idxs = feeds_fetches_info.fetches_mlvalue_idxs;
  ExecutionFrame frame{feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state};
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  const auto& graph_viewer = session_state.GetGraphViewer();
  int num_trials = 10;

  LOGS(logger, WARNING) << "Profiling graph...";

  for (const auto& node_exec_plan : exec_plan_vec) {
    auto node_index = node_exec_plan.node_index;
    const auto& node = *graph_viewer.GetNode(node_exec_plan.node_index);
    auto p_op_kernel = session_state.GetKernel(node_index);
    OpKernelContextInternal op_kernel_context(session_state, frame, *p_op_kernel, logger, false /* terminate_flag */);

    if (!node.Name().compare("")) {
      // TODO: Handle nodes with no name
      LOGS(logger, WARNING) << "Running node with no name of type " << node.OpType() << "...";
      ORT_RETURN_IF_ERROR(RunOp(p_op_kernel, op_kernel_context, node, logger));
    } else if (op_runtimes_.find(node.Name()) == op_runtimes_.end()) {
      std::vector<float> runtimes;
      for (int i = 0; i < num_trials; i++) {
        LOGS(logger, WARNING) << "[" << i+1 << "/" << num_trials << "] Profiling node " << node.Name() << " (Op type " << node.OpType() << ")...";
        auto begin = std::chrono::high_resolution_clock::now();
        ORT_RETURN_IF_ERROR(RunOp(p_op_kernel, op_kernel_context, node, logger));
        // TODO: Synchronize
        auto end = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        runtimes.push_back(std::chrono::duration<float>(runtime).count());
      }
      std::sort(runtimes.begin(), runtimes.end());
      op_runtimes_[node.Name()] = runtimes[runtimes.size() / 2];

      LOGS(logger, WARNING) << "Finished profiling node " << node.Name() << ", runtime is " << op_runtimes_[node.Name()] << " microseconds";
    } else {
      LOGS(logger, WARNING) << "Already profiled node " << node.Name();
    }
  }
  return Status::OK();
}

Status OpRuntimeProfiler::RunOp(const OpKernel* p_op_kernel, OpKernelContextInternal& op_kernel_context, const Node& node, const logging::Logger& logger) {
  Status compute_status;

  ORT_TRY {
    compute_status = p_op_kernel->Compute(&op_kernel_context);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      compute_status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
    });
  }

  if (!compute_status.IsOK()) {
    std::ostringstream ss;
    ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
       << "' Status Message: " << compute_status.ErrorMessage();
    const auto msg_string = ss.str();
    LOGS(logger, ERROR) << msg_string;
    return Status(compute_status.Category(), compute_status.Code(), msg_string);
  }
  return compute_status;
}

float OpRuntimeProfiler::GetRuntime(const std::string& name) {
  if (op_runtimes_.find(name) == op_runtimes_.end()) {
    return -1;
  } else {
    return op_runtimes_[name];
  }
}

} // namespace onnxruntime
