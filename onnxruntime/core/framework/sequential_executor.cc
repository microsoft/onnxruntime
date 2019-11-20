// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sequential_executor.h"

#include <chrono>
#include <thread>
#include <vector>
#include <sstream>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/utils.h"

// Define this symbol to create Concurrency Visualizer markers.
// See https://docs.microsoft.com/en-us/visualstudio/profiling/concurrency-visualizer-sdk
// You will need to install Concurrency Visualizer and add the SDK to the project that compiles this file
// via Analyze->Concurrency Visualizer->Add SDK to Project...
// #define CONCURRENCY_VISUALIZER
#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
using namespace Concurrency;
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include <Windows.h>
#include "core/platform/tracing.h"
namespace {
LARGE_INTEGER OrtGetPerformanceFrequency() {
  LARGE_INTEGER v;
  // On systems that run Windows XP or later, the QueryPerformanceFrequency function will always succeed
  // and will thus never return zero.
  (void)QueryPerformanceFrequency(&v);
  return v;
}

LARGE_INTEGER perf_freq = OrtGetPerformanceFrequency();
}  // namespace
#endif

namespace onnxruntime {

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger);

Status SequentialExecutor::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger) {
  const bool is_profiler_enabled = session_state.Profiler().IsEnabled();
  TimePoint tp;
  TimePoint sync_time_begin;
  TimePoint kernel_begin_time;

  if (is_profiler_enabled) {
    tp = session_state.Profiler().StartTime();
  }

  ExecutionFrame frame{feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state};

  LOGS(logger, INFO) << "Begin execution";
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

  // uncomment the line below to dump execution plan
  //std::cout << std::make_pair(p_seq_exec_plan, &session_state) << "\n";
  const auto* graph_viewer = session_state.GetGraphViewer();

#ifdef CONCURRENCY_VISUALIZER
  // need unique name for the series. number of nodes should be good enough for a subgraph
  char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
  if (graph_viewer->IsSubgraph()) {
    auto s = graph_viewer->ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
    std::copy(s.cbegin(), s.cend(), series_name);
  }

  diagnostic::marker_series series(series_name);
#endif

  for (const auto& node_exec_plan : exec_plan_vec) {
    if (terminate_flag_) {
      LOGS(logger, WARNING) << "Exiting due to terminate flag being set to true.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
    }

    auto node_index = node_exec_plan.node_index;
    const auto& node = *graph_viewer->GetNode(node_exec_plan.node_index);

#ifdef CONCURRENCY_VISUALIZER
    series.write_flag(node.Name().c_str());
#endif

    auto p_op_kernel = session_state.GetKernel(node_index);

    // if a kernel has been added in the session state, it better be NON-null.
    if (p_op_kernel == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ",
                             node.Name());
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_start;
    QueryPerformanceCounter(&kernel_start);
#endif
    // construct OpKernelContext
    // TODO: log kernel inputs?
    OpKernelContextInternal op_kernel_context(session_state, frame, *p_op_kernel, logger, terminate_flag_);
    // TODO: log kernel outputs?
    if (is_profiler_enabled) {
      sync_time_begin = session_state.Profiler().StartTime();
    }

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
#if defined DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeInputs(op_kernel_context, p_op_kernel->Node());
#endif

    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_fence_before",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});

      // call compute on the kernel
      VLOGS(logger, 1) << "Computing kernel: " << p_op_kernel->Node().Name();

      kernel_begin_time = session_state.Profiler().StartTime();
    }

#ifdef CONCURRENCY_VISUALIZER
    {
      diagnostic::span span(series, "%s.%d", node.OpType().c_str(), node.Index());
#endif
      Status compute_status;

      try {
        compute_status = p_op_kernel->Compute(&op_kernel_context);
      } catch (const std::exception& ex) {
        compute_status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      }

      if (!compute_status.IsOK()) {
        std::ostringstream ss;
        ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
           << "' Status Message: " << compute_status.ErrorMessage();
        const auto msg_string = ss.str();
        LOGS(logger, ERROR) << msg_string;
        return Status(compute_status.Category(), compute_status.Code(), msg_string);
      }

#ifdef CONCURRENCY_VISUALIZER
    }
#endif

    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_kernel_time",
                                                     kernel_begin_time,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}, {"provider", p_op_kernel->KernelDef().Provider()}});

      sync_time_begin = session_state.Profiler().StartTime();
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
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_stop;
    QueryPerformanceCounter(&kernel_stop);
    LARGE_INTEGER elapsed;
    elapsed.QuadPart = kernel_stop.QuadPart - kernel_start.QuadPart;
    elapsed.QuadPart *= 1000000;
    elapsed.QuadPart /= perf_freq.QuadPart;
    // Log an event
    TraceLoggingWrite(telemetry_provider_handle,  // handle to my provider
                      "OpEnd",       // Event Name that should uniquely identify your event.
                      TraceLoggingValue(p_op_kernel->KernelDef().OpName().c_str(), "op_name"),
                      TraceLoggingValue(elapsed.QuadPart, "time"));
#endif
    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     p_op_kernel->Node().Name() + "_fence_after",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
    }

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
    utils::DumpNodeOutputs(op_kernel_context, p_op_kernel->Node(), session_state);
#endif

    // free ml-values corresponding to this node
    VLOGS(logger, 1) << "Releasing node ML values after computing kernel: " << p_op_kernel->Node().Name();
    ORT_RETURN_IF_ERROR(ReleaseNodeMLValues(frame, seq_exec_plan, node_exec_plan, logger));
  }

  VLOGS(logger, 1) << "Fetching output.";
  // ExecutionFrame::Finalize will update 'fetches' with the final output
  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));
  VLOGS(logger, 1) << "Done with execution.";

  if (frame.HasMemoryPatternPlanner()) {
    std::vector<std::reference_wrapper<const TensorShape>> input_shapes;
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
      auto& tensor = feed.Get<Tensor>();
      input_shapes.push_back(std::cref(tensor.Shape()));
    }

    if (all_tensors) {
      auto mem_patterns = onnxruntime::make_unique<MemoryPatternGroup>();
      ORT_RETURN_IF_ERROR(frame.GeneratePatterns(mem_patterns.get()));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(input_shapes, std::move(mem_patterns)));
    }
  }

  if (is_profiler_enabled) {
    session_state.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", tp);
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
