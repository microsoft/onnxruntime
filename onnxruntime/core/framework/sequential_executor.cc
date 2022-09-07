// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sequential_executor.h"

#include <chrono>
#include <thread>
#include <vector>
#include <sstream>

#include <string>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/utils.h"
//#include "core/framework/session_scope.h"
#include "core/framework/session_scope_profiler.h"

namespace onnxruntime {

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger);

Status SequentialExecutor::Execute(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger) {

  ExecutionFrame frame{feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state};
  SessProfiler sess_profiler(session_state, frame);

#if !defined(ORT_MINIMAL_BUILD)
  const auto* const to_be_executed_nodes = session_state.GetToBeExecutedNodes(fetch_mlvalue_idxs);
  const bool only_execute_path_to_fetches = only_execute_path_to_fetches_ && (to_be_executed_nodes != nullptr);

  if (only_execute_path_to_fetches) {
    VLOGS(logger, 1) << to_be_executed_nodes->size() << " nodes to be executed\n";
  }
#else
  ORT_UNUSED_PARAMETER(only_execute_path_to_fetches_);
#endif

  LOGS(logger, INFO) << "Begin execution";
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

  const auto& graph_viewer = session_state.GetGraphViewer();

  for (const auto& node_exec_plan : exec_plan_vec) {
    if (terminate_flag_) {
      LOGS(logger, WARNING) << "Exiting due to terminate flag being set to true.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
    }

    auto node_index = node_exec_plan.node_index;

#if !defined(ORT_MINIMAL_BUILD)
    // If it is not necessary to execute the node.
    if (only_execute_path_to_fetches && to_be_executed_nodes->count(node_index) == 0) {
      continue;
    }
#endif

    const auto& node = *graph_viewer.GetNode(node_exec_plan.node_index);
    auto p_op_kernel = session_state.GetKernel(node_index);

    // if a kernel has been added in the session state, it better be NON-null.
    if (p_op_kernel == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ",
                             node.Name());

    // construct OpKernelContext
    OpKernelContextInternal op_kernel_context(session_state, frame, *p_op_kernel, logger, terminate_flag_);

    // sync before compute
    int queue_id = p_op_kernel->KernelDef().ExecQueueId();
    if (seq_exec_plan.NodeHasFence(node_index)) {
      for (int input_index = 0; input_index < op_kernel_context.InputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.InputFence(input_index);
        if (fence) {
          auto execution_provider_type = p_op_kernel->Node().GetExecutionProviderType();
          if (OrtMemTypeCPUInput == p_op_kernel->KernelDef().InputMemoryType(input_index)) {
            execution_provider_type = kCpuExecutionProvider;
          }
          fence->BeforeUsingAsInput(execution_provider_type, queue_id);
        }
      }

      for (int input_index = 0; input_index < op_kernel_context.ImplicitInputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.ImplicitInputFence(input_index);
        if (fence) {
          auto execution_provider_type = p_op_kernel->Node().GetExecutionProviderType();
          if (OrtMemTypeCPUInput == p_op_kernel->KernelDef().InputMemoryType(input_index)) {
            execution_provider_type = kCpuExecutionProvider;
          }
          fence->BeforeUsingAsInput(execution_provider_type, queue_id);
        }
      }

      for (int output_index = 0; output_index < op_kernel_context.OutputCount(); ++output_index) {
        Fence_t fence = op_kernel_context.OutputFence(output_index);
        if (fence) {
          fence->BeforeUsingAsOutput(p_op_kernel->Node().GetExecutionProviderType(), queue_id);
        }
      }
    }

    Status compute_status;
    {
      KernelProfiler kernel_profiler(sess_profiler, op_kernel_context, *p_op_kernel);

      ORT_TRY {
#ifdef ENABLE_TRAINING
        if (p_op_kernel->KernelDef().AllocateInputsContiguously()) {
          ORT_RETURN_IF_ERROR(utils::VerifyInputTensorsAllocatedContiguously(&op_kernel_context));
        }
#endif
        compute_status = p_op_kernel->Compute(&op_kernel_context);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          compute_status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
        });
      }
    }

    if (!compute_status.IsOK()) {
      std::ostringstream ss;
      ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
         << "' Status Message: " << compute_status.ErrorMessage();
      const auto msg_string = ss.str();
      LOGS(logger, ERROR) << msg_string;
      return Status(compute_status.Category(), compute_status.Code(), msg_string);
    }

    // sync after compute for outputs
    if (seq_exec_plan.NodeHasFence(node_index)) {
      for (int input_index = 0; input_index < op_kernel_context.InputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.InputFence(input_index);
        if (fence) {
          fence->AfterUsedAsInput(queue_id);
        }
      }

      for (int input_index = 0; input_index < op_kernel_context.ImplicitInputCount(); ++input_index) {
        Fence_t fence = op_kernel_context.ImplicitInputFence(input_index);
        if (fence) {
          fence->AfterUsedAsInput(queue_id);
        }
      }

      for (int output_index = 0; output_index < op_kernel_context.OutputCount(); ++output_index) {
        Fence_t fence = op_kernel_context.OutputFence(output_index);
        if (fence) {
          fence->AfterUsedAsOutput(queue_id);
        }
      }
    }

    // free ml-values corresponding to this node
    VLOGS(logger, 1) << "Releasing node ML values.";
    ORT_RETURN_IF_ERROR(ReleaseNodeMLValues(frame, seq_exec_plan, node_exec_plan, logger));
  }

  VLOGS(logger, 1) << "Fetching output.";
  // ExecutionFrame::Finalize will update 'fetches' with the final output
  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));
  VLOGS(logger, 1) << "Done with execution.";

  if (frame.HasMemoryPatternPlanner()) {
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
    }

    if (all_tensors) {
      MemoryPatternGroup mem_patterns;
      ORT_RETURN_IF_ERROR(frame.GeneratePatterns(mem_patterns));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(feeds, std::move(mem_patterns)));
    }
  }
  return Status::OK();
}

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger) {
  for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
    auto ort_value_idx = seq_exec_plan.to_be_freed[i];
    VLOGS(logger, 1) << "Releasing ort_value with index: " << ort_value_idx;
    ORT_RETURN_IF_ERROR(frame.ReleaseMLValue(ort_value_idx));
  }

  return Status::OK();
}
}  // namespace onnxruntime
