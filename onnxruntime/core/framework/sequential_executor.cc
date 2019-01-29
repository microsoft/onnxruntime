// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sequential_executor.h"

#include <chrono>
#include <thread>
#include <vector>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {

static Status FetchOutput(const MLValueNameIdxMap& name_idx_map,
                          ExecutionFrame& frame,
                          const std::vector<std::string>& output_names,
                          std::vector<MLValue>& fetches,
                          const logging::Logger& logger);

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger);

Status SequentialExecutor::Execute(const SessionState& session_state,
                                   const NameMLValMap& feeds,
                                   const std::vector<std::string>& output_names,
                                   std::vector<MLValue>& fetches,
                                   const std::unordered_map<size_t, CustomAllocator> fetch_allocators,
                                   const logging::Logger& logger) {
  bool f_profiler_enabled = session_state.Profiler().FEnabled();
  TimePoint tp;
  TimePoint sync_time_begin;
  TimePoint kernel_begin_time;

  if (f_profiler_enabled) {
    tp = session_state.Profiler().StartTime();
  }

  ExecutionFrame frame{feeds, output_names, fetches, fetch_allocators, session_state};

  LOGS(logger, INFO) << "Begin execution";
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

  // uncomment the line below to dump execution plan
  //std::cout << std::make_pair(p_seq_exec_plan, &session_state) << "\n";

  for (const auto& node_exec_plan : exec_plan_vec) {
    if (terminate_flag_) {
      LOGS(logger, WARNING) << "Exiting due to terminate flag being set to true.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
    }

    auto node_index = node_exec_plan.node_index;
    auto p_op_kernel = session_state.GetKernel(node_index);

    // if a kernel has been added in the session state, it better be NON-null.
    if (p_op_kernel == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ",
                             session_state.GetGraphViewer()->GetNode(node_index)->Name());

    // construct OpKernelContext
    // TODO: log kernel inputs?
    OpKernelContextInternal op_kernel_context(frame, *p_op_kernel, logger, p_op_kernel->Node().ImplicitInputDefs(),
                                              terminate_flag_);
    // TODO: log kernel outputs?
    if (f_profiler_enabled) {
      sync_time_begin = session_state.Profiler().StartTime();
    }

    // sync before compute
    int queue_id = p_op_kernel->KernelDef().ExecQueueId();
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

    if (f_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_fence_before",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});

      // call compute on the kernel
      VLOGS(logger, 1) << "Computing kernel: " << p_op_kernel->Node().Name();

      kernel_begin_time = session_state.Profiler().StartTime();
    }
    ORT_RETURN_IF_ERROR(p_op_kernel->Compute(&op_kernel_context));

    if (f_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_kernel_time",
                                                     kernel_begin_time,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});

      sync_time_begin = session_state.Profiler().StartTime();
    }

    // sync after compute for outputs
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

    if (f_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_fence_after",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
    }

    // free ml-values corresponding to this node
    VLOGS(logger, 1) << "Releasing node ML values after computing kernel: " << p_op_kernel->Node().Name();
    ORT_RETURN_IF_ERROR(ReleaseNodeMLValues(frame, seq_exec_plan, node_exec_plan, logger));
  }

  VLOGS(logger, 1) << "Fetching output.";
  ORT_RETURN_IF_ERROR(FetchOutput(session_state.GetMLValueNameIdxMap(), frame, output_names, fetches, logger));

  if (frame.HasPlan()) {
    std::vector<TensorShape> input_shapes;
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.second.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.second.Get<Tensor>();
      input_shapes.push_back(tensor.Shape());
    }

    if (all_tensors) {
      auto mem_patterns = std::make_unique<MemoryPatternGroup>();
      ORT_RETURN_IF_ERROR(frame.GeneratePatterns(mem_patterns.get()));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(input_shapes, std::move(mem_patterns)));
    }
  }

  if (f_profiler_enabled) {
    session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", tp);
  }

  return Status::OK();
}

static Status FetchOutput(const MLValueNameIdxMap& name_idx_map,
                          ExecutionFrame& frame,
                          const std::vector<std::string>& output_names,
                          std::vector<MLValue>& fetches,
                          const logging::Logger& logger) {
  if (fetches.empty()) {
    fetches.resize(output_names.size());
  } else {
    // this should've been checked before already
    ORT_ENFORCE(output_names.size() == fetches.size(),
                "output_names vector size: " + std::to_string(output_names.size()) +
                    " does not match that of fetches vector: " + std::to_string(fetches.size()));
  }

  auto idx = 0;

  for (const auto& oname : output_names) {
    VLOGS(logger, 1) << "Attempting to fetch output with name: " << oname;
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(name_idx_map.GetIdx(oname, mlvalue_index));
    const MLValue& output_mlvalue = frame.GetMLValue(mlvalue_index);
    VLOGS(logger, 1) << "Copying fetched MLValue to output vector";
    fetches[idx++] = output_mlvalue;
  }

  VLOGS(logger, 1) << "Done with execution.";
  return Status::OK();
}

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger) {
  for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
    auto mlvalue_idx = seq_exec_plan.to_be_freed[i];
    VLOGS(logger, 1) << "Releasing mlvalue with index: " << mlvalue_idx;
    ORT_RETURN_IF_ERROR(frame.ReleaseMLValue(mlvalue_idx));
  }

  return Status::OK();
}
}  // namespace onnxruntime
