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

// #define TRACE_EXECUTION

namespace onnxruntime {

static Status ReleaseNodeMLValues(ExecutionFrame& frame,
                                  const SequentialExecutionPlan& seq_exec_plan,
                                  const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan,
                                  const logging::Logger& logger);

static std::unordered_set<NodeIndex> CalculateToBeExecutedNodes(const std::vector<int>& fetch_mlvalue_idxs,
                                                                const GraphViewer& graph_viewer,
                                                                const OrtValueNameIdxMap& mlvalue_name_idxs) {
  // Get the nodes generating the fetches.
  std::vector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  for (auto idx : fetch_mlvalue_idxs) {
    std::string node_arg_name;
    if (!mlvalue_name_idxs.GetName(idx, node_arg_name).IsOK()) {
      return {};
    }

    auto ending_node = graph_viewer.GetGraph()->GetProducerNode(node_arg_name);
    nodes.push_back(ending_node);
  }

  // Reversely traverse to get reachable nodes.
  std::unordered_set<NodeIndex> reachable_nodes;
  graph_viewer.GetGraph()->ReverseDFSFrom(
      nodes, {}, [&reachable_nodes](const Node* n) { reachable_nodes.insert(n->Index()); });

  return reachable_nodes;
}

Status SequentialExecutor::Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                   const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger) {
  const bool is_profiler_enabled = session_state.Profiler().IsEnabled();
  TimePoint tp;
  TimePoint sync_time_begin;
  TimePoint kernel_begin_time;
  size_t input_activation_sizes = 0;
  size_t input_parameter_sizes = 0;
  size_t total_output_sizes = 0;

  if (is_profiler_enabled) {
    tp = session_state.Profiler().StartTime();
  }

  ExecutionFrame frame{feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, session_state};

  std::unordered_set<NodeIndex> to_be_executed_nodes;
  if (only_execute_path_to_fetches_) {
    // TODO: This information could be potentially stored in a limited size cache.
    to_be_executed_nodes = CalculateToBeExecutedNodes(fetch_mlvalue_idxs,
                                                      *session_state.GetGraphViewer(),
                                                      session_state.GetOrtValueNameIdxMap());
    VLOGS(logger, 1) << to_be_executed_nodes.size() << " nodes to be executed\n";
  }

  LOGS(logger, INFO) << "Begin execution";
  const SequentialExecutionPlan& seq_exec_plan = *session_state.GetExecutionPlan();
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

  // Enable TRACE_EXECUTION compile flag to dump execution plan
#if defined(TRACE_EXECUTION)
  std::cout << std::make_pair(&seq_exec_plan, &session_state) << std::endl;
#endif

  for (const auto& node_exec_plan : exec_plan_vec) {
    if (terminate_flag_) {
      LOGS(logger, WARNING) << "Exiting due to terminate flag being set to true.";
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
    }

    auto node_index = node_exec_plan.node_index;

    // If it is not necessary to execute the node.
    if (only_execute_path_to_fetches_ && to_be_executed_nodes.count(node_index) == 0) {
      continue;
    }

    auto p_op_kernel = session_state.GetKernel(node_index);

    // if a kernel has been added in the session state, it better be NON-null.
    if (p_op_kernel == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Got nullptr from GetKernel for node: ",
                             session_state.GetGraphViewer()->GetNode(node_index)->Name());

    std::string node_name = p_op_kernel->Node().Name();
    if (node_name.empty()) {
      // Node name field is often blank in execution graph, so derive something meaningful for profile traces and logs.
      node_name = p_op_kernel->Node().OpType() + "_" + std::to_string(node_index);
    }

    // construct OpKernelContext
    // TODO: log kernel inputs?
    OpKernelContextInternal op_kernel_context(session_state, frame, *p_op_kernel, logger,
                                              p_op_kernel->Node().ImplicitInputDefs(), terminate_flag_);
    // TODO: log kernel outputs?
    if (is_profiler_enabled) {
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

#if defined DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeInputs(op_kernel_context, p_op_kernel->Node());
#endif

    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name + "_fence_before",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});

      // call compute on the kernel
      VLOGS(logger, 1) << "Computing kernel: " << node_name;

      kernel_begin_time = session_state.Profiler().StartTime();
    }

    if (is_profiler_enabled) {
      // Calculate total input sizes for this operation.
      input_activation_sizes = 0;
      input_parameter_sizes = 0;
      for (auto i = 0; i < op_kernel_context.InputCount(); i++) {
        const OrtValue* p_input = op_kernel_context.GetInputMLValue(i);
        if (p_input->IsTensor()) {
          const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
          const Tensor* p_tensor = nullptr;
          bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
          if (!is_param) {
            p_tensor = &(p_input->Get<Tensor>());
          }
          size_t tensor_size = p_tensor->Size();
#if defined(TRACE_EXECUTION)
          TensorShape tensor_shape = p_tensor->Shape();
          size_t element_size = p_tensor->DataType()->Size();
          std::cout << node_name << " input[" << i << "]"
                    << " is_param=" << is_param
                    << " size=" << tensor_size
                    << " shape=" << tensor_shape.ToString()
                    << " element_size=" << element_size
                    << "\n";
#endif
          if (is_param) {
            input_parameter_sizes += tensor_size;
          }
          else {
            input_activation_sizes += tensor_size;
          }
        }
      }
    }

    const auto& compute_status = p_op_kernel->Compute(&op_kernel_context);
    if (!compute_status.IsOK()) {
      std::ostringstream ss;
      ss << "Non-zero status code returned while running Node: " << node_name
         << " Status Message: " << compute_status.ErrorMessage();
      const auto msg_string = ss.str();
      LOGS(logger, ERROR) << msg_string;
      return Status(compute_status.Category(), compute_status.Code(), msg_string);
    }

    if (is_profiler_enabled) {

      // Calculate total output sizes for this operation.
      total_output_sizes = 0;
      for (auto i = 0; i < op_kernel_context.OutputCount(); i++) {
        const OrtValue* p_output = op_kernel_context.GetOutputMLValue(i);
        if (p_output->IsTensor()) {
          const auto& tensor = p_output->Get<Tensor>();
          size_t tensor_size = tensor.Size();
#if defined(TRACE_EXECUTION)
          const TensorShape& tensor_shape = tensor.Shape();
          std::cout << node_name << " output[" << i << "]"
            << " size=" << tensor_size
            << " shape=" << tensor_shape.ToString()
            << " element_size=" << tensor.DataType()->Size()
            << "\n";
#endif
          total_output_sizes += tensor_size;
        }
      }

#if defined(TRACE_EXECUTION)
      // Trace execution step.
      const Node& node = p_op_kernel->Node();
      std::cout << "Executed op kernel node " << node_name
                << " Index=" << node.Index()
                << " OpType=" << node.OpType()
                << " Name=" << node.Name()
                << " Activation_Size=" << input_activation_sizes
                << " Parameter_Size=" << input_parameter_sizes
                << " Output_Size=" << total_output_sizes
                << "\n";
#endif

      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name + "_kernel_time",
                                                     kernel_begin_time,
                                                     // Log additional operation args / info.
                                                     {
                                                         {"op_name", p_op_kernel->KernelDef().OpName()},
                                                         {"provider", p_op_kernel->KernelDef().Provider()},
                                                         {"graph_index", std::to_string(p_op_kernel->Node().Index())},
                                                         {"exec_plan_index", std::to_string(node_index)},
                                                         {"activation_size", std::to_string(input_activation_sizes)},
                                                         {"parameter_size", std::to_string(input_parameter_sizes)},
                                                         {"output_size", std::to_string(total_output_sizes)},
                                                     });

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

    if (is_profiler_enabled) {
      session_state.Profiler().EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                                     node_name + "_fence_after",
                                                     sync_time_begin,
                                                     {{"op_name", p_op_kernel->KernelDef().OpName()}});
    }

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
    utils::DumpNodeOutputs(op_kernel_context, p_op_kernel->Node(), session_state);
#endif

    // free ml-values corresponding to this node
    VLOGS(logger, 1) << "Releasing node ML values after computing kernel: " << node_name;
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
      auto mem_patterns = std::make_unique<MemoryPatternGroup>();
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
