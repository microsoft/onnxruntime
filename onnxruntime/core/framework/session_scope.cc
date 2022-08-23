// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "session_scope.h"
#include "core/common/profiler.h"
#include "core/framework/memory_info.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel_context_internal.h"
#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
#endif
#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include <Windows.h>
#include "core/platform/tracing.h"
#endif
#if defined DEBUG_NODE_INPUTS_OUTPUTS
#include "core/framework/debug_node_inputs_outputs_utils.h"
#endif
#include <thread>

namespace onnxruntime {

using namespace onnxruntime::profiling;

#ifdef ORT_MINIMAL_BUILD

class SessionScopeImpl {
};

SessionScope::SessionScope(const SessionState&, const ExecutionFrame&) {}
SessionScope::~SessionScope() {}

class KernelScopeImpl {
};

KernelScope::KernelScope(OpKernelContextInternal&, const OpKernel&, SessionScope&) {}
KernelScope::~KernelScope() {}

#else

static void CalculateTotalOutputSizes(OpKernelContextInternal* op_kernel_context,
                                      size_t& total_output_sizes,
                                      const std::string& node_name,
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
                                     size_t& input_activation_sizes,
                                     size_t& input_parameter_sizes,
                                     const std::string& node_name,
                                     std::string& input_type_shape) {
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

#ifdef CONCURRENCY_VISUALIZER
using namespace Concurrency;
class ConcurrencyKernelScope;
class ConcurrencySessScope {
 public:
  friend class ConcurrencyKernelScope;
  ConcurrencySessScope(const GraphViewer& graph_viewer) : series_(ComposeSeriesName(graph_viewer)) {}

 private:
  static std::string ComposeSeriesName(const GraphViewer& graph_viewer) {
    char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
    if (graph_viewer.IsSubgraph()) {
      auto s = graph_viewer.ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
      std::copy(s.cbegin(), s.cend(), series_name);
    }
    return series_name;
  }
  diagnostic::marker_series series_;
};

class ConcurrencyKernelScope {
 public:
  ConcurrencyKernelScope(ConcurrencySessScope& scope,
                         const Node& node) : span_(scope.series_,
                                                   "%s.%d",
                                                   node.OpType().c_str(),
                                                   node.Index()) {
    scope.series_.write_flag(node.Name().c_str());
  };

 private:
  diagnostic::span span_;
};
#else
class ConcurrencySessScope {
 public:
  ConcurrencySessScope(const GraphViewer&){};
};

class ConcurrencyKernelScope {
 public:
  ConcurrencyKernelScope(ConcurrencySessScope&, const Node&){};
};
#endif

#ifdef ENABLE_NVTX_PROFILE
class NVTXKernelScope;
class NVTXSessScope {
 public:
  friend class NVTXKernelScope;
  NVTXSessScope(const std::thread::id& thread_id) : sess_tag_(profile::Context::GetInstance().GetThreadTagOrDefault(thread_id)),
                                                    forward_range_("Batch-" + sess_tag_ + " Forward", profile::Color::White),
                                                    backward_range_("Batch-" + sess_tag_ + " Backward", profile::Color::Black) {}
  ~NVTXSessScope() {
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
  }

 private:
  const std::string sess_tag_;
  profile::NvtxRangeCreator forward_range_;
  profile::NvtxRangeCreator backward_range_;
};
class NVTXKernelScope {
 public:
  NVTXKernelScope(NVTXSessScope& nvtx_sess_scope,
                  const onnxruntime::Node& node) : node_(node),
                                                   node_compute_range_(MakeString(node_.OpType(), ".",
                                                                                  node_.Index(), "(",
                                                                                  node_.Name(), ")"),
                                                                       profile::Color::Yellow) {
    if (node_.Description() != "Backward pass" &&
        !nvtx_sess_scope.forward_range_.IsBeginCalled()) {
      // Start timing forward pass when encountering the first forward node.
      nvtx_sess_scope.forward_range_.Begin();
    } else if (node.Description() == "Backward pass" &&
               !nvtx_sess_scope.backward_range_.IsBeginCalled() &&
               nvtx_sess_scope.forward_range_.IsBeginCalled()) {
      // Start timing backward pass when encountering the first backward node.
      // In the meanwhile, forward range ends.
      nvtx_sess_scope.forward_range_.End();
      nvtx_sess_scope.backward_range_.Begin();
    }
    node_compute_range_.Begin();
  }
  ~NVTXKernelScope() {
    node_compute_range_.End();
  }

 private:
  const onnxruntime::Node& node_;
  profile::NvtxRangeCreator node_compute_range_;
};
#else
class NVTXSessScope {
 public:
  NVTXSessScope(const std::thread::id&) {}
};
class NVTXKernelScope {
 public:
  NVTXKernelScope(NVTXSessScope&, const onnxruntime::Node&) {}
};
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
class DumpKernelScope {
 public:
  DumpKernelScope(const SessionState& sess_state,
                  OpKernelContextInternal& context,
                  const Node& node,
                  size_t iteration) : sess_state_(sess_state),
                                      context_(context),
                                      node_(node),
                                      dump_ctx_{iteration, node_.Index()} {
    utils::DumpNodeInputs(dump_ctx_, context_, node_, sess_state_);
  }

  ~DumpKernelScope() {
    utils::DumpNodeOutputs(dump_ctx_, context_, node_, sess_state_);
  }

 private:
  const SessionState& sess_state_;
  OpKernelContextInternal& context_;
  const Node& node_;
  utils::NodeDumpContext dump_ctx_;
};
#else
class DumpKernelScope {
 public:
  DumpKernelScope(const SessionState&,
                  OpKernelContextInternal&,
                  const Node&,
                  size_t) {}
};
#endif

#ifdef ORT_MEMORY_PROFILE
class MemSessScope {
 public:
  MemSessScope(const SessionState& sess_state,
               const ExecutionFrame& frame) : logger_(sess_state.Logger()),
                                              frame_(frame),
                                              profiler_(*sess_state.GetMemoryProfiler()) {}
  ~MemSessScope() {
    profiler_.CreateEvents(
        "dynamic activations_" + std::to_string(profiler_.GetMemoryInfo().GetIteration()),
        profiler_.GetAndIncreasePid(),
        MemoryInfo::MapType::DynamicActivation, "", 0);
    profiler_.Clear();
    for (auto i : frame_.GetStaticMemorySizeInfo()) {
      LOGS(logger_, INFO) << "[Memory] ExecutionFrame statically allocates "
                          << i.second << " bytes for " << i.first << std::endl;
    }

    for (auto i : frame_.GetDynamicMemorySizeInfo()) {
      LOGS(logger_, INFO) << "[Memory] ExecutionFrame dynamically allocates "
                          << i.second << " bytes for " << i.first << std::endl;
    }
  }

 private:
  const logging::Logger& logger_;
  const ExecutionFrame& frame_;
  MemoryProfiler& profiler_;
};
#else
class MemSessScope {
 public:
  MemSessScope(const SessionState&,
               const ExecutionFrame&) {}
};
#endif

class ProfilerSessScope {
 public:
  ProfilerSessScope(Profiler& profiler) : profiler_(profiler) {
    enabled_ = profiler_.IsEnabled();
    if (enabled_) {
      sess_start_ = profiler_.Start();
    }
  }
  ~ProfilerSessScope() {
    if (enabled_) {
      profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", sess_start_);
    }
  }
  bool Enabled() const { return enabled_; }
  Profiler& profiler_;

 private:
  bool enabled_;
  TimePoint sess_start_;
};

#ifdef TRACE_EXECUTION
class TraceSessScope {
 public:
  TraceSessScope(const SessionState& sess_state) {
    const auto& seq_exec_plan = sess_state.GetExecutionPlan();
    std::cout << std::make_pair(&seq_exec_plan, &sess_state) << std::endl;
  }
};
class TraceKernelScope {
 public:
  TraceKernelScope(const OpKernelContextInternal& context,
                   const onnxruntime::OpKernel& kernel) : context_(context), kernel_(kernel) {
    const int input_count = context_.InputCount();
    for (auto i = 0; i < input_count; i++) {
      const OrtValue* p_input = context_.GetInputMLValue(i);
      if (p_input && p_input->IsTensor()) {
        const OpKernelInfo& op_kernel_info = kernel_.Info();
        const Tensor* p_tensor = nullptr;
        bool is_param = op_kernel_info.TryGetConstantInput(i, &p_tensor);
        if (!is_param) {
          p_tensor = &(p_input->Get<Tensor>());
        }
        size_t tensor_size = p_tensor->SizeInBytes();
        const TensorShape& tensor_shape = p_tensor->Shape();
        size_t element_size = p_tensor->DataType()->Size();
        std::cout << kernel_.Node().Name() << " input[" << i << "]"
                  << " is_param=" << is_param
                  << " size=" << tensor_size
                  << " shape=" << tensor_shape.ToString()
                  << " element_size=" << element_size
                  << std::endl;
      }
    }
  }
  ~TraceKernelScope() {
    int output_count = context_.OutputCount();
    for (auto i = 0; i < output_count; i++) {
      const OrtValue* p_output = context_.GetOutputMLValue(i);
      if (p_output != nullptr && p_output->IsTensor()) {
        const auto& tensor = p_output->Get<Tensor>();
        size_t tensor_size = tensor.SizeInBytes();
        const TensorShape& tensor_shape = tensor.Shape();
        std::cout << node_name << " output[" << i << "]"
                  << " size=" << tensor_size
                  << " shape=" << tensor_shape.ToString()
                  << " element_size=" << tensor.DataType()->Size()
                  << std::endl;
      }
    }
    auto& node = kernel_.Node();
    std::cout << "Executed op kernel node " << node.Name()
              << " Index=" << node.Index()
              << " OpType=" << node.OpType()
              << " Name=" << node.Name()
              << std::endl;
  }

 private:
  const OpKernelContextInternal& context_;
  const onnxruntime::OpKernel& kernel_;
};
#else
class TraceSessScope {
 public:
  TraceSessScope(const SessionState&){};
};
class TraceKernelScope {
 public:
  TraceKernelScope(const OpKernelContextInternal&,
                   const onnxruntime::OpKernel&) {}
};
#endif

class SessionScopeImpl {
 public:
  friend class KernelScopeImpl;
  SessionScopeImpl(const SessionState& sess_state,
                   const ExecutionFrame& frame) : sess_state_(sess_state),
                                                  frame_(frame),
                                                  trace_scope_(sess_state_),
                                                  concurrency_scope_(sess_state_.GetGraphViewer()),
                                                  nvtx_scope_(std::this_thread::get_id()),
                                                  profiler_scope_(sess_state_.Profiler()),
                                                  mem_scope_(sess_state_, frame_) {
    iteration_++;
  }
  ~SessionScopeImpl() = default;

 private:
  std::atomic<size_t> iteration_{0};
  const SessionState& sess_state_;
  const ExecutionFrame& frame_;
  TraceSessScope trace_scope_;
  ConcurrencySessScope concurrency_scope_;
  NVTXSessScope nvtx_scope_;
  ProfilerSessScope profiler_scope_;
  MemSessScope mem_scope_;
  //add new session scope here
};

SessionScope::SessionScope(const SessionState& sess_state, const ExecutionFrame& frame) {
  impl_ = std::make_unique<SessionScopeImpl>(sess_state, frame);
}

SessionScope::~SessionScope() {}

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
LARGE_INTEGER OrtGetPerformanceFrequency() {
  LARGE_INTEGER v;
  // On systems that run Windows XP or later, the QueryPerformanceFrequency function will always succeed
  // and will thus never return zero.
  (void)QueryPerformanceFrequency(&v);
  return v;
}

LARGE_INTEGER perf_freq = OrtGetPerformanceFrequency();

class InstrumentKernelScope {
 public:
  InstrumentKernelScope(const OpKernel& kernel) : kernel_(kernel) {
    QueryPerformanceCounter(&kernel_start_);
  }
  ~InstrumentKernelScope() {
    LARGE_INTEGER kernel_stop;
    QueryPerformanceCounter(&kernel_stop);
    LARGE_INTEGER elapsed;
    elapsed.QuadPart = kernel_stop.QuadPart - kernel_start_.QuadPart;
    elapsed.QuadPart *= 1000000;
    elapsed.QuadPart /= perf_freq.QuadPart;
    // Log an event
    TraceLoggingWrite(telemetry_provider_handle,  // handle to my provider
                      "OpEnd",                    // Event Name that should uniquely identify your event.
                      TraceLoggingValue(kernel_.KernelDef().OpName().c_str(), "op_name"),
                      TraceLoggingValue(elapsed.QuadPart, "time"));
  }

 private:
  const OpKernel& kernel_;
  LARGE_INTEGER kernel_start_;
};
#else
class InstrumentKernelScope {
 public:
  InstrumentKernelScope(const OpKernel&){};
};
#endif

class KernelScopeImpl {
 public:
  KernelScopeImpl(OpKernelContextInternal& context,
                  const OpKernel& kernel,
                  SessionScope& sess_scope) : context_(context),
                                              kernel_(kernel),
                                              sess_scope_(sess_scope),
                                              sess_state_(sess_scope.impl_->sess_state_),
                                              concur_scope_(sess_scope.impl_->concurrency_scope_, kernel.Node()),
                                              nvtx_scope_(sess_scope.impl_->nvtx_scope_, kernel.Node()),
                                              dump_scope_(sess_scope.impl_->sess_state_, context, kernel.Node(), sess_scope.impl_->iteration_),
                                              //mem_scope_(sess_scope.impl_->sess_state_),
                                              trace_scope_(context, kernel),
                                              instrument_scope_(kernel) {
    is_profiler_enabled_ = sess_scope.impl_->profiler_scope_.Enabled();

    if (is_profiler_enabled_) {
      auto& node = kernel.Node();
      node_name_ = node.Name().empty() ? MakeString(node.OpType(), "_", node.Index()) : node.Name();
      auto& profiler = sess_scope.impl_->profiler_scope_.profiler_;
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_before",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});

      concurrency::ThreadPool::StartProfiling(sess_state_.GetThreadPool());
      kernel_begin_time_ = profiler.Start();
      CalculateTotalInputSizes(&context,
                               &kernel,
                               input_activation_sizes_,
                               input_parameter_sizes_,
                               node_name_,
                               input_type_shape_);
    }
  }

  ~KernelScopeImpl() {
    if (is_profiler_enabled_) {
      CalculateTotalOutputSizes(&context_, total_output_sizes_, node_name_, output_type_shape_);
      auto& profiler = sess_scope_.impl_->profiler_scope_.profiler_;
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_kernel_time",
                                     kernel_begin_time_,
                                     {
                                         {"op_name", kernel_.KernelDef().OpName()},
                                         {"provider", kernel_.KernelDef().Provider()},
                                         {"node_index", std::to_string(kernel_.Node().Index())},
                                         {"activation_size", std::to_string(input_activation_sizes_)},
                                         {"parameter_size", std::to_string(input_parameter_sizes_)},
                                         {"output_size", std::to_string(total_output_sizes_)},
                                         {"input_type_shape", input_type_shape_},
                                         {"output_type_shape", output_type_shape_},
                                         {"thread_scheduling_stats", concurrency::ThreadPool::StopProfiling(sess_state_.GetThreadPool())},
                                     });
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_after",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});
    }
  }

 private:
  bool is_profiler_enabled_ = false;
  TimePoint kernel_begin_time_;
  std::string node_name_;
  OpKernelContextInternal& context_;
  const OpKernel& kernel_;
  SessionScope& sess_scope_;
  const SessionState& sess_state_;
  size_t input_activation_sizes_{};
  size_t input_parameter_sizes_{};
  size_t total_output_sizes_{};
  std::string input_type_shape_{};
  std::string output_type_shape_{};
  ConcurrencyKernelScope concur_scope_;
  NVTXKernelScope nvtx_scope_;
  DumpKernelScope dump_scope_;
  TraceKernelScope trace_scope_;
  InstrumentKernelScope instrument_scope_;
  //add new kernel scope here
};

KernelScope::KernelScope(OpKernelContextInternal& kernel_context,
                         const OpKernel& kernel, SessionScope& sess_scope) {
  impl_ = std::make_unique<KernelScopeImpl>(kernel_context, kernel, sess_scope);
}

KernelScope::~KernelScope() {}

#endif

}  // namespace onnxruntime