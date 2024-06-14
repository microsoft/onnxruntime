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
#include "core/framework/stream_execution_context.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/utils.h"

#if defined DEBUG_NODE_INPUTS_OUTPUTS
#include "core/framework/debug_node_inputs_outputs_utils.h"
#endif

#ifdef ENABLE_NVTX_PROFILE
// This header is for profile using Nvidia's visual profilier.
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif

// #define TRACE_EXECUTION

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

static void CalculateTotalOutputSizes(OpKernelContextInternal* op_kernel_context,
                                      size_t& total_output_sizes, const std::string& node_name,
                                      std::string& output_type_shape) {
  // Calculate total output sizes for this operation.
  std::stringstream ss;
  int added_type_shapes = 0;
  ss << "[";
  total_output_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  int output_count = op_kernel_context->OutputCount();
  for (auto i = 0; i < output_count; i++) {
    const OrtValue* p_output = op_kernel_context->GetOutputMLValue(i);
    if (p_output != nullptr && p_output->IsTensor()) {
      const auto& tensor = p_output->Get<Tensor>();
      size_t tensor_size = tensor.SizeInBytes();
#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = tensor.Shape();
      std::cout << node_name << " output[" << i << "]"
                << " size=" << tensor_size
                << " shape=" << tensor_shape.ToString()
                << " element_size=" << tensor.DataType()->Size()
                << "\n";
#endif
      total_output_sizes += tensor_size;
      auto shape_str = tensor.Shape().ToString();
      ss << (added_type_shapes++ > 0 ? "," : "")
         << "{\"" << DataTypeImpl::ToString(tensor.DataType()) << "\":["
         << shape_str.substr(1, shape_str.size() - 2) << "]}";
    }
  }
  ss << "]";
  output_type_shape = ss.str();
}

static void CalculateTotalInputSizes(const OpKernelContextInternal* op_kernel_context,
                                     const onnxruntime::OpKernel* p_op_kernel,
                                     size_t& input_activation_sizes, size_t& input_parameter_sizes,
                                     const std::string& node_name, std::string& input_type_shape) {
  // Calculate total input sizes for this operation.
  std::stringstream ss;
  ss << "[";
  int added_type_shapes = 0;
  input_activation_sizes = 0;
  input_parameter_sizes = 0;
  ORT_UNUSED_PARAMETER(node_name);
  const int input_count = op_kernel_context->InputCount();
  for (auto i = 0; i < input_count; i++) {
    const OrtValue* p_input = op_kernel_context->GetInputMLValue(i);
    if (p_input != nullptr && p_input->IsTensor()) {
      const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
      const Tensor* p_tensor = nullptr;
      bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
      if (!is_param) {
        p_tensor = &(p_input->Get<Tensor>());
      }
      size_t tensor_size = p_tensor->SizeInBytes();

#if defined(TRACE_EXECUTION)
      const TensorShape& tensor_shape = p_tensor->Shape();
      size_t element_size = p_tensor->DataType()->Size();
      LOGS(logger, INFO) << node_name << " input[" << i << "]"
                         << " is_param=" << is_param
                         << " size=" << tensor_size
                         << " shape=" << tensor_shape.ToString()
                         << " element_size=" << element_size
                         << "\n";
#endif
      if (is_param) {
        input_parameter_sizes += tensor_size;
      } else {
        input_activation_sizes += tensor_size;
      }
      auto shape_str = p_tensor->Shape().ToString();
      ss << (added_type_shapes++ > 0 ? "," : "")
         << "{\"" << DataTypeImpl::ToString(p_tensor->DataType()) << "\":["
         << shape_str.substr(1, shape_str.size() - 2) << "]}";
    }
  }
  ss << "]";
  input_type_shape = ss.str();
}

class KernelScope;

#ifdef CONCURRENCY_VISUALIZER
std::string ComposeSeriesName(const GraphViewer& graph_viewer) {
  char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
  if (graph_viewer.IsSubgraph()) {
    auto s = graph_viewer.ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
    std::copy(s.cbegin(), s.cend(), series_name);
  }
  return series_name;
}
#endif

class SessionScope {
 public:
  friend class KernelScope;
  SessionScope(const SessionState& session_state, const ExecutionFrame& frame)
      : session_state_(session_state)
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
        ,
        frame_(frame)
#endif
#ifdef CONCURRENCY_VISUALIZER
        ,
        series_(ComposeSeriesName(session_state.GetGraphViewer()))
#endif
#ifdef ENABLE_NVTX_PROFILE
        ,
        session_tag_(profile::Context::GetInstance().GetThreadTagOrDefault(std::this_thread::get_id())),
        forward_range_("Batch-" + session_tag_ + " Forward", profile::Color::White),
        backward_range_("Batch-" + session_tag_ + " Backward", profile::Color::Black)
#endif
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
        ,
        dump_context_{
            session_state_.GetGraphExecutionCounter(), 0}
#endif
  {
    if (session_state_.Profiler().IsEnabled()) {
      session_start_ = session_state.Profiler().Start();
    }

    auto& logger = session_state_.Logger();
    VLOGS(logger, 0) << "Begin execution";
    const SequentialExecutionPlan& seq_exec_plan = *session_state_.GetExecutionPlan();
    const auto& exec_plan_vec = seq_exec_plan.execution_plan;
    VLOGS(logger, 1) << "Size of execution plan vector: " << exec_plan_vec.size();

// Enable TRACE_EXECUTION compile flag to dump execution plan
#if defined(TRACE_EXECUTION)
    std::cout << std::make_pair(&seq_exec_plan, &session_state) << std::endl;
#endif
#if defined(ORT_MINIMAL_BUILD) || !defined(ORT_MEMORY_PROFILE)
    ORT_UNUSED_PARAMETER(frame);
#endif
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionScope);

  ~SessionScope() {
#ifdef ENABLE_NVTX_PROFILE
    // Make sure forward Range object call Begin and End.
    if (!forward_range_.IsBeginCalled()) {
      forward_range_.Begin();
    }
    if (!forward_range_.IsEndCalled()) {
      forward_range_.End();
    }
    // Make sure backward Range object call Begin and End.
    if (!backward_range_.IsBeginCalled()) {
      backward_range_.Begin();
    }
    if (!backward_range_.IsEndCalled()) {
      backward_range_.End();
    }
#endif

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    if (flush_memory_info_) {
      session_state_.GetMemoryProfiler()->CreateEvents(
          "dynamic activations_" + std::to_string(session_state_.GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
          session_state_.GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::DynamicActivation, "", 0);
      session_state_.GetMemoryProfiler()->Clear();
    }
#endif

    if (session_state_.Profiler().IsEnabled()) {
      session_state_.Profiler().EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", session_start_);
    }
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    auto& logger = session_state_.Logger();
    for (auto i : frame_.GetStaticMemorySizeInfo()) {
      LOGS(logger, VERBOSE) << "[Memory] ExecutionFrame statically allocates "
                            << i.second << " bytes for " << i.first << std::endl;
    }

    for (auto i : frame_.GetDynamicMemorySizeInfo()) {
      LOGS(logger, VERBOSE) << "[Memory] ExecutionFrame dynamically allocates "
                            << i.second << " bytes for " << i.first << std::endl;
    }
#endif
  }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  void SetFlushMemoryInfoFlag(bool flush_memory_info) {
    flush_memory_info_ = flush_memory_info;
  }
#endif

 private:
  const SessionState& session_state_;
  TimePoint session_start_;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  const ExecutionFrame& frame_;
  // Whether memory profiler need create events and flush to file.
  // For partial graph run, when the last subgraph of the whole graph is executing, we need flush to file.
  bool flush_memory_info_ = true;
#endif

#ifdef CONCURRENCY_VISUALIZER
  diagnostic::marker_series series_;
#endif

#ifdef ENABLE_NVTX_PROFILE
  const std::string session_tag_;
  profile::NvtxRangeCreator forward_range_;
  profile::NvtxRangeCreator backward_range_;
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  utils::NodeDumpContext dump_context_;
#endif
};

class KernelScope {
 public:
  KernelScope(SessionScope& session_scope,
              OpKernelContextInternal& kernel_context,
              const OpKernel& kernel)
      : session_scope_(session_scope),
        session_state_(session_scope_.session_state_),
        kernel_context_(kernel_context),
        kernel_(kernel)
#ifdef CONCURRENCY_VISUALIZER
        ,
        span_(session_scope_.series_, "%s.%d", kernel_.Node().OpType().c_str(), kernel_.Node().Index())
#endif
#ifdef ENABLE_NVTX_PROFILE
        ,
        node_compute_range_(MakeString(kernel_.Node().OpType(),
                                       ".",
                                       kernel_.Node().Index(),
                                       "(",
                                       kernel_.Node().Name(),
                                       ")"),
                            profile::Color::Yellow)
#endif
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
        ,
        dump_context_{
            session_scope_.dump_context_.iteration, kernel_.Node().Index()}
#endif
  {
#ifdef CONCURRENCY_VISUALIZER
    session_scope_.series_.write_flag(kernel_.Node().Name().c_str());
#endif

#ifdef ENABLE_NVTX_PROFILE
    {
      auto& node = kernel_.Node();
      profile::NvtxRangeCreator& forward_range = session_scope_.forward_range_;
      profile::NvtxRangeCreator& backward_range = session_scope_.backward_range_;
      if (node.Description() != "Backward pass" && !forward_range.IsBeginCalled()) {
        // Start timing forward pass when encountering the first forward node.
        forward_range.Begin();
      } else if (node.Description() == "Backward pass" && !backward_range.IsBeginCalled() &&
                 forward_range.IsBeginCalled()) {
        // Start timing backward pass when encountering the first backward node.
        // In the meanwhile, forward range ends.
        forward_range.End();
        backward_range.Begin();
      }
    }
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    LARGE_INTEGER kernel_start;
    QueryPerformanceCounter(&kernel_start);
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeInputs(dump_context_, kernel_context_, kernel_.Node(), session_state_);
#endif

#ifdef ENABLE_NVTX_PROFILE
    node_compute_range_.Begin();
#endif

    if (session_state_.Profiler().IsEnabled()) {
      auto& node = kernel.Node();
      node_name_ = node.Name().empty() ? MakeString(node.OpType(), "_", node.Index()) : node.Name();
      auto& profiler = session_state_.Profiler();
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_before",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});
      concurrency::ThreadPool::StartProfiling(session_state_.GetThreadPool());
      VLOGS(session_state_.Logger(), 1) << "Computing kernel: " << node_name_;
      kernel_begin_time_ = session_state_.Profiler().Start();
      CalculateTotalInputSizes(&kernel_context, &kernel_,
                               input_activation_sizes_, input_parameter_sizes_,
                               node_name_, input_type_shape_);
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(KernelScope);

  ~KernelScope() {
#ifdef ENABLE_NVTX_PROFILE
    node_compute_range_.End();
#endif

    if (session_state_.Profiler().IsEnabled()) {
      auto& profiler = session_state_.Profiler();
      std::string output_type_shape_;
      CalculateTotalOutputSizes(&kernel_context_, total_output_sizes_, node_name_, output_type_shape_);
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_kernel_time",
                                     kernel_begin_time_,
                                     // Log additional operation args / info.
                                     {
                                         {"op_name", kernel_.KernelDef().OpName()},
                                         {"provider", kernel_.KernelDef().Provider()},
                                         {"node_index", std::to_string(kernel_.Node().Index())},
                                         {"activation_size", std::to_string(input_activation_sizes_)},
                                         {"parameter_size", std::to_string(input_parameter_sizes_)},
                                         {"output_size", std::to_string(total_output_sizes_)},
                                         {"input_type_shape", input_type_shape_},
                                         {"output_type_shape", output_type_shape_},
                                         {"thread_scheduling_stats",
                                          concurrency::ThreadPool::StopProfiling(session_state_.GetThreadPool())},
                                     });
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_after",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});
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
                      "OpEnd",                    // Event Name that should uniquely identify your event.
                      TraceLoggingValue(p_op_kernel->KernelDef().OpName().c_str(), "op_name"),
                      TraceLoggingValue(elapsed.QuadPart, "time"));
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    utils::DumpNodeOutputs(dump_context_, kernel_context_, kernel_.Node(), session_state_);
#endif
  }  //~KernelScope

 private:
  TimePoint kernel_begin_time_;
  SessionScope& session_scope_;
  const SessionState& session_state_;
  std::string node_name_;
  OpKernelContextInternal& kernel_context_;
  const OpKernel& kernel_;

  size_t input_activation_sizes_{};
  size_t input_parameter_sizes_{};
  size_t total_output_sizes_{};
  std::string input_type_shape_;

#ifdef CONCURRENCY_VISUALIZER
  diagnostic::span span_;
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator node_compute_range_;
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  utils::NodeDumpContext dump_context_;
#endif
};

onnxruntime::Status ExecuteKernel(StreamExecutionContext& ctx,
                                  NodeIndex idx,
                                  size_t stream_idx,
                                  const bool& terminate_flag,
                                  SessionScope& session_scope) {
  auto* p_kernel = ctx.GetSessionState().GetKernel(idx);
  if (p_kernel->KernelDef().OpName() == "YieldOp") {
    // Do not execute YieldOp (it is an no-op anyways).
    // Decrement the reference count of tensors that are not needed beyond this point.
    // REVEIW(codemzs): The current model assumes the intermediate tensors that are exported
    // as graph outputs are owned by ORT, the risk of caller freeing the tensor or manipulating tensor
    // memory lingers while the tensor is used downstream after the export.
    ctx.RecycleNodeInputs(idx);
    return Status::OK();
  }
  // TODO: set terminate flag from run_option
  OpKernelContextInternal kernel_ctx(ctx.GetSessionState(),
                                     ctx.GetExecutionFrame(),
                                     *p_kernel,
                                     ctx.GetLogger(),
                                     terminate_flag,
                                     ctx.GetDeviceStream(stream_idx));
  onnxruntime::Status status;
  auto& logger = ctx.GetLogger();
  if (p_kernel->IsAsync()) {
    ORT_THROW("Async Kernel Support is not implemented yet.");
  } else {
    KernelScope kernel_scope(session_scope, kernel_ctx, *p_kernel);
    ORT_TRY {
#ifdef ENABLE_TRAINING
      // AllocateInputsContiguously - is only required for NCCL kernels
      // can be moved under USE_NCCL
      if (p_kernel->KernelDef().AllocateInputsContiguously()) {
        ORT_RETURN_IF_ERROR(utils::VerifyInputTensorsAllocatedContiguously(&kernel_ctx));
      }

      // This is most probably deprecated code and is causing unnecessary complexity.
      // Can be removed.
      // Cache lookup. Currently we only cache single-output nodes,
      // to keep memory overhead impact in check. Hence we only look in cache
      // if the current node has one output.
      bool reuse_cached_value = false;
      std::string cached_arg_name;
      auto& cache = ctx.GetOrtValueCache();
      if (cache != nullptr) {
        if (p_kernel->Node().OutputDefs().size() == 1) {
          cached_arg_name = p_kernel->Node().OutputDefs()[0]->Name();
          if (cache.get()->count(cached_arg_name)) {  // found arg in cache_
            VLOGS(logger, 1) << "Found OrtValue in cache for arg: " << cached_arg_name;
            reuse_cached_value = true;
          }
        }
      }
      if (!reuse_cached_value) {
        status = p_kernel->Compute(&kernel_ctx);
      } else {
        status = kernel_ctx.SetOutputMLValue(0, cache.get()->at(cached_arg_name));
      }
#else
      status = p_kernel->Compute(&kernel_ctx);
#endif
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what());
      });
    }
  }
  if (!status.IsOK()) {
    std::ostringstream ss;
    const auto& node = p_kernel->Node();
    ss << "Non-zero status code returned while running " << node.OpType() << " node. Name:'" << node.Name()
       << "' Status Message: " << status.ErrorMessage();
    // If the computation failed, we still can record the memory consumption
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    ctx.GetSessionState().GetMemoryProfiler()->CreateEvents(
        "dynamic activations_" + std::to_string(ctx.GetSessionState().GetMemoryProfiler()->GetMemoryInfo().GetIteration()),
        ctx.GetSessionState().GetMemoryProfiler()->GetAndIncreasePid(), MemoryInfo::MapType::DynamicActivation, "", 0);
#endif
    const auto msg_string = ss.str();
    LOGS(logger, ERROR) << msg_string;
    return Status(status.Category(), status.Code(), msg_string);
  }
  ctx.RecycleNodeInputs(idx);
  VLOGS(logger, 0) << "stream " << stream_idx << " launch kernel with idx " << idx;
  return Status::OK();
}

onnxruntime::Status ExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                                   const DeviceStreamCollection* device_streams,
#endif
                                   const bool& terminate_flag,
                                   const bool only_execute_path_to_fetches,
                                   bool single_thread_mode) {
  auto* execution_plan = session_state.GetExecutionPlan();
  VLOGS(logger, 0) << "Number of streams: " << execution_plan->execution_plan.size();
  int32_t valid_streams = 0;
  for (auto& stream : execution_plan->execution_plan) {
    if (stream && stream->steps_.size() > 0)
      valid_streams++;
  }

  // prepare the execution context, notifications got initialized.
#ifdef ORT_ENABLE_STREAM
  StreamExecutionContext ctx(session_state,
                             valid_streams,
                             execution_plan->notification_owners,
                             execution_plan->num_barriers,
                             device_streams,
                             feed_mlvalue_idxs,
                             feeds,
                             fetch_mlvalue_idxs,
                             fetches,
                             fetch_allocators,
                             logger,
                             single_thread_mode);
#else
  StreamExecutionContext ctx(session_state,
                             valid_streams,
                             feed_mlvalue_idxs,
                             feeds,
                             fetch_mlvalue_idxs,
                             fetches,
                             fetch_allocators,
                             logger,
                             single_thread_mode);
#endif
#ifdef ENABLE_TRAINING
  if (only_execute_path_to_fetches) {
    auto* node_to_execute = session_state.GetToBeExecutedRange(fetch_mlvalue_idxs);
    ctx.SetNodeToExecute(node_to_execute);
  }
#else
  ORT_UNUSED_PARAMETER(only_execute_path_to_fetches);
#endif

  SessionScope session_scope(session_state, ctx.GetExecutionFrame());

  auto* tp = single_thread_mode ? nullptr : session_state.GetInterOpThreadPool();

  for (size_t i = 0; i < execution_plan->execution_plan.size(); ++i) {
    if (execution_plan->execution_plan[i]->steps_.empty()) {
      // execution context is initialized with number of valid streams
      // for invalid stream (0 steps), it doesn't count in number of tasks
      // so don't need to invoke CompleteTask here
      // ctx.CompleteTask();
    } else {
      concurrency::ThreadPool::Schedule(tp, [i, &ctx, &terminate_flag, &session_scope]() {
        RunSince(i, ctx, session_scope, terminate_flag, 0);
      });
    }
  }

  ctx.WaitAll();
  ORT_RETURN_IF_ERROR(ctx.TaskStatus());
  ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GetOutputs(fetches));
  if (ctx.GetExecutionFrame().HasMemoryPatternPlanner()) {
    bool all_tensors = true;
    for (const auto& feed : feeds) {
      if (!(feed.IsTensor())) {
        all_tensors = false;
        break;
      }
    }

    if (all_tensors) {
      MemoryPatternGroup mem_patterns;
      ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GeneratePatterns(mem_patterns));
      ORT_RETURN_IF_ERROR(session_state.UpdateMemoryPatternGroupCache(feeds, std::move(mem_patterns)));
    }
  }

  return Status::OK();
}

#ifdef ENABLE_TRAINING
onnxruntime::Status PartialExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                          gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                          std::vector<OrtValue>& fetches,
                                          const std::unordered_map<size_t, IExecutor::CustomAllocator>&
                                              fetch_allocators,
                                          const logging::Logger& logger,
                                          const DeviceStreamCollection* device_streams,
                                          const bool& terminate_flag,
                                          bool single_thread_mode,
                                          PartialGraphExecutionState& state,
                                          const OrtValueCachePtr& cache,
                                          int32_t partial_graph_index) {
  auto& ctx = state.GetExecutionContext(feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches,
                                        fetch_allocators, session_state, logger, device_streams);
  auto* plan = session_state.GetExecutionPlan();

  ctx.SetCurrentRange(&state.GetProgramRegions(session_state));

  SessionScope session_scope(session_state, ctx.GetExecutionFrame());

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  // Only flush memory info for the 2nd partial graph execution (since ORTModule runs this function twice).
  session_scope.SetFlushMemoryInfoFlag(partial_graph_index == 1);
#else
  ORT_UNUSED_PARAMETER(partial_graph_index);
#endif

  ctx.SetOrtValueCache(cache);

  auto* tp = single_thread_mode ? nullptr : session_state.GetInterOpThreadPool();

  for (size_t i = 0; i < plan->execution_plan.size(); ++i) {
    if (!plan->execution_plan[i]->steps_.empty()) {
      concurrency::ThreadPool::Schedule(tp, [i, &ctx, &terminate_flag, &session_scope]() {
        auto* range = ctx.GetCurrentRange();
        size_t start = !range ? 0 : range->stream_pc_range[i].first;
        RunSince(i, ctx, session_scope, terminate_flag, start);
      });
    }
  }

  if (!single_thread_mode) {
    ctx.WaitAll();
  }

  ORT_RETURN_IF_ERROR(ctx.TaskStatus());
  ORT_RETURN_IF_ERROR(ctx.GetExecutionFrame().GetOutputs(fetches));
  return Status::OK();
}
#endif

}  // namespace onnxruntime
