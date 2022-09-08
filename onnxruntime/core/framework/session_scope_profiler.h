// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
#endif
#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif
#if defined DEBUG_NODE_INPUTS_OUTPUTS
#include "core/framework/debug_node_inputs_outputs_utils.h"
#endif
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include <Windows.h>
#include "core/platform/tracing.h"
#endif

namespace onnxruntime {

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

struct ISessProfiler {
 protected:
  ISessProfiler(const SessionState& sess_state,
                const ExecutionFrame& frame) : sess_state_(sess_state),
                                               frame_(frame) {}

 public:
  virtual ISessProfiler& GetAnotherSessProfiler() {
    ORT_ENFORCE(false, "must return upstream profiler");
    return *this;
  }
  const SessionState& sess_state_;
  const ExecutionFrame& frame_;
};

struct IKernelProfiler {
 protected:
  IKernelProfiler(ISessProfiler& sess_profiler,
                  const OpKernelContextInternal& context,
                  const OpKernel& kernel) : sess_profiler_(sess_profiler),
                                            context_(context),
                                            kernel_(kernel) {}
  ISessProfiler& sess_profiler_;
  const OpKernelContextInternal& context_;
  const OpKernel& kernel_;
};

///////////////////////////////////////////////// CONCURRENCY /////////////////////////////////////////////////
#ifdef CONCURRENCY_VISUALIZER
using namespace Concurrency;
//for session profiling
struct ConcurrencySessProfiler : public ISessProfiler {
  ConcurrencySessProfiler(const SessionState& sess_state,
                          const ExecutionFrame& frame) : public ISessProfiler(sess_state, frame),
                                                         series_(ComposeSeriesName(sess_state.GetGraphViewer()))
#ifdef ANOTHER_SESS_PROFILER
                                                         ,
                                                         another_sess_profiler_(sess_state, frame)
#endif
  {
  }

  static std::string ComposeSeriesName(const GraphViewer& graph_viewer) {
    char series_name[MaxSeriesNameLengthInChars] = "MainGraph";
    if (graph_viewer.IsSubgraph()) {
      auto s = graph_viewer.ParentNode()->Name().substr(0, MaxSeriesNameLengthInChars - 1);
      std::copy(s.cbegin(), s.cend(), series_name);
    }
    return series_name;
  }

  diagnostic::marker_series series_;

#ifdef ANOTHER_SESS_PROFILER
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER ConcurrencySessProfiler
//for kernel profiling
struct ConcurrencyKernelProfiler : public IKernelProfiler {
  ConcurrencyKernelProfiler(ISessProfiler& sess_profiler,
                            const OpKernelContextInternal& context,
                            const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
                                                      series_(dynamic_cast<ConcurrencySessProfiler&>(sess_profiler).series_),
                                                      span_(series_, "%s.%d",
                                                            kernel.Node().OpType().c_str(),
                                                            kernel.Node().Index())
#ifdef ANOTHER_KERNEL_PROFILER
                                                      ,
                                                      another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel);
#endif
  {
    scope.series_.write_flag(kernel.Node().Name().c_str());
  }

  diagnostic::marker_series& series_;
  diagnostic::span span_;

#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER ConcurrencyKernelProfiler
#endif

////////////////////////////////////////////////////// NVTX //////////////////////////////////////////////////////
#ifdef ENABLE_NVTX_PROFILE
//for session profiling
struct NVTXSessProfiler : public ISessProfiler {
  NVTXSessProfiler(const SessionState& sess_state,
                   const ExecutionFrame& frame) : ISessProfiler(sess_state, frame),
                                                  sess_tag_(profile::Context::GetInstance().GetThreadTagOrDefault(std::this_thread::get_id())),
                                                  forward_range_("Batch-" + sess_tag_ + " Forward", profile::Color::White),
                                                  backward_range_("Batch-" + sess_tag_ + " Backward", profile::Color::Black)
#ifdef ANOTHER_SESS_PROFILER
                                                  ,
                                                  another_sess_profiler_(sess_state, frame)
#endif
  {
  }
  ~NVTXSessProfiler() {
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
  const std::string sess_tag_;
  profile::NvtxRangeCreator forward_range_;
  profile::NvtxRangeCreator backward_range_;
#ifdef ANOTHER_SESS_PROFILER
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER NVTXSessProfiler
//for kernel profiling
struct NVTXKernelProfiler : public IKernelProfiler {
  NVTXKernelProfiler(ISessProfiler& sess_profiler,
                     const OpKernelContextInternal& context,
                     const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
                                               sess_profiler_(dynamic_cast<NVTXSessProfiler&>(sess_profiler)),
                                               node_compute_range_(MakeString(kernel.Node().OpType(), ".",
                                                                              kernel.Node().Index(), "(",
                                                                              kernel.Node().Name(), ")"),
                                                                   profile::Color::Yellow)
#ifdef ANOTHER_KERNEL_PROFILER
                                               ,
                                               another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
    auto& node = kernel.Node();
    if (node.Description() != "Backward pass" &&
        !sess_profiler_.forward_range_.IsBeginCalled()) {
      sess_profiler_.forward_range_.Begin();
    } else if (node.Description() == "Backward pass" &&
               !sess_profiler_.backward_range_.IsBeginCalled() &&
               sess_profiler_.forward_range_.IsBeginCalled()) {
      sess_profiler_.forward_range_.End();
      sess_profiler_.backward_range_.Begin();
    }
    node_compute_range_.Begin();
  }
  ~NVTXKernelProfiler() {
    node_compute_range_.End();
  }
  NVTXSessProfiler& sess_profiler_;
  profile::NvtxRangeCreator node_compute_range_;
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER NVTXKernelProfiler
#endif
///////////////////////////////////////// DUMP INPUT AND OUTPUT ///////////////////////////////////////////

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
//for session profiling
struct DumpSessProfiler : public ISessProfiler {
  DumpSessProfiler(const SessionState& sess_state,
                   const ExecutionFrame& frame) : ISessProfiler(sess_state, frame)
#ifdef ANOTHER_SESS_PROFILER
                                                  ,
                                                  another_sess_profiler_(sess_state, frame)
#endif
  {
    iteration++;
  }
#ifdef ANOTHER_SESS_PROFILER
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
  static std::atomic_size_t iteration;
};
std::atomic_size_t DumpSessProfiler::iteration = {0};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER DumpSessProfiler
//for kernel profiling
struct DumpKernelProfiler : public IKernelProfiler {
  DumpKernelProfiler(ISessProfiler& sess_profiler,
                     const OpKernelContextInternal& context,
                     const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
                                               sess_profiler_(dynamic_cast<DumpSessProfiler&>(sess_profiler)),
                                               dump_ctx_ { DumpSessProfiler::iteration, kernel.Node().Index() }
#ifdef ANOTHER_KERNEL_PROFILER
  ,
      another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
    utils::DumpNodeInputs(dump_ctx_, context, kernel.Node(), sess_profiler.sess_state_);
  }
  ~DumpKernelProfiler() {
    utils::DumpNodeOutputs(dump_ctx_, const_cast<OpKernelContextInternal&>(context_), kernel_.Node(), sess_profiler_.sess_state_);
  }
  DumpSessProfiler& sess_profiler_;
  utils::NodeDumpContext dump_ctx_;
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER DumpKernelProfiler
#endif

//////////////////////////////////////////// MEM ////////////////////////////////////////////

#ifdef ORT_MEMORY_PROFILE
//for session profiling
struct MemSessProfiler : public ISessProfiler {
  MemSessProfiler(const SessionState& sess_state,
                  const ExecutionFrame& frame) : ISessProfiler(sess_state, frame),
                                                 logger_(sess_state.Logger()),
                                                 profiler_(*sess_state.GetMemoryProfiler())
#ifdef ANOTHER_SESS_PROFILER
                                                 ,
                                                 another_sess_profiler_(sess_state, frame)
#endif
  {
  }
  ~MemSessProfiler() {
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

  const logging::Logger& logger_;
  MemoryProfiler& profiler_;

#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER MemSessProfiler
//for kernel profiling
struct MemKernelProfiler : public IKernelProfiler {
  MemKernelProfiler(ISessProfiler& sess_profiler,
                    const OpKernelContextInternal& context,
                    const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel)
#ifdef ANOTHER_KERNEL_PROFILER
                                              ,
                                              another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
  }
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER MemKernelProfiler
#endif

////////////////////////////////////////////////// PERF //////////////////////////////////////////////////
//for session profiling
struct PerfSessProfiler : public ISessProfiler {
  PerfSessProfiler(const SessionState& sess_state,
                   const ExecutionFrame& frame) : ISessProfiler(sess_state, frame),
                                                  profiler_(sess_state.Profiler())
#ifdef ANOTHER_SESS_PROFILER
                                                  ,
                                                  another_sess_profiler_(sess_state, frame)
#endif
  {
    enabled_ = profiler_.IsEnabled();
    if (enabled_) {
      sess_start_ = profiler_.Start();
    }
  }
  ~PerfSessProfiler() {
    if (enabled_) {
      profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", sess_start_);
    }
  }
  bool Enabled() const { return enabled_; }
  profiling::Profiler& profiler_;
  bool enabled_;
  TimePoint sess_start_;
#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER PerfSessProfiler
//for kernel profiling
struct PerfKernelProfiler : public IKernelProfiler {
  PerfKernelProfiler(ISessProfiler& sess_profiler,
                     const OpKernelContextInternal& context,
                     const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
                                               perf_sess_profiler_(dynamic_cast<PerfSessProfiler&>(sess_profiler))
#ifdef ANOTHER_KERNEL_PROFILER
                                               ,
                                               another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
    if (perf_sess_profiler_.profiler_.IsEnabled()) {
      auto& node = kernel.Node();
      node_name_ = node.Name().empty() ? MakeString(node.OpType(), "_", node.Index()) : node.Name();
      auto& profiler = perf_sess_profiler_.profiler_;
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_before",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});

      concurrency::ThreadPool::StartProfiling(sess_profiler.sess_state_.GetThreadPool());
      kernel_begin_time_ = profiler.Start();
      CalculateTotalInputSizes(&context,
                               &kernel,
                               input_activation_sizes_,
                               input_parameter_sizes_,
                               node_name_,
                               input_type_shape_);
    }
  }
  ~PerfKernelProfiler() {
    if (perf_sess_profiler_.profiler_.IsEnabled()) {
      CalculateTotalOutputSizes(const_cast<OpKernelContextInternal*>(&context_), total_output_sizes_, node_name_, output_type_shape_);
      auto& profiler = perf_sess_profiler_.profiler_;
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
                                         {"thread_scheduling_stats", concurrency::ThreadPool::StopProfiling(sess_profiler_.sess_state_.GetThreadPool())},
                                     });
      auto sync_time_begin = profiler.Start();
      profiler.EndTimeAndRecordEvent(profiling::NODE_EVENT,
                                     node_name_ + "_fence_after",
                                     sync_time_begin,
                                     {{"op_name", kernel_.KernelDef().OpName()}});
    }
  }
  PerfSessProfiler& perf_sess_profiler_;
  std::string node_name_;
  TimePoint kernel_begin_time_;
  size_t input_activation_sizes_{};
  size_t input_parameter_sizes_{};
  size_t total_output_sizes_{};
  std::string input_type_shape_{};
  std::string output_type_shape_{};
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER PerfKernelProfiler

////////////////////////////////////////////////// TRACE //////////////////////////////////////////////////

#ifdef TRACE_EXECUTION
//for session profiling
class TraceSessProfiler : public ISessProfiler {
 public:
  TraceSessProfiler(const SessionState& sess_state,
                    const ExecutionFrame& frame) : ISessProfiler(sess_state, frame)
#ifdef ANOTHER_SESS_PROFILER
                                                   ,
                                                   another_sess_profiler_(sess_state, frame)
#endif

  {
    const auto& seq_exec_plan = sess_state_.GetExecutionPlan();
    std::cout << std::make_pair(&seq_exec_plan, &sess_state_) << std::endl;
  }
#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER TraceSessProfiler
//for kernel profiling
struct TraceKernelProfiler : public IKernelProfiler {
  TraceKernelProfiler(ISessProfiler& sess_profiler,
                     const OpKernelContextInternal& context,
                     const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel),
#ifdef ANOTHER_KERNEL_PROFILER
                                               ,
                                               another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
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
  ~TraceKernelProfiler() {
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
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER TraceKernelProfiler
#endif

////////////////////////////////////////////////// INSTRUMENT //////////////////////////////////////////////////
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
LARGE_INTEGER OrtGetPerformanceFrequency() {
  LARGE_INTEGER v;
  // On systems that run Windows XP or later, the QueryPerformanceFrequency function will always succeed
  // and will thus never return zero.
  (void)QueryPerformanceFrequency(&v);
  return v;
}

LARGE_INTEGER perf_freq = OrtGetPerformanceFrequency();
//for session profiling
class InstrumentSessProfiler : public ISessProfiler {
 public:
  InstrumentSessProfiler(const SessionState& sess_state,
                         const ExecutionFrame& frame) : ISessProfiler(sess_state, frame)
#ifdef ANOTHER_SESS_PROFILER
                                                        ,
                                                        another_sess_profiler_(sess_state, frame)
#endif
  {
  }
#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER InstrumentSessProfiler
//for kernel profiling
struct InstrumentKernelProfiler : public IKernelProfiler {
  InstrumentKernelProfiler(ISessProfiler& sess_profiler,
                           const OpKernelContextInternal& context,
                           const OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel)
#ifdef ANOTHER_KERNEL_PROFILER
                                                     ,
                                                     another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
    QueryPerformanceCounter(&kernel_start_);
  }
  ~InstrumentKernelProfiler() {
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
  LARGE_INTEGER kernel_start_;
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};
#undef ANOTHER_KERNEL_PROFILER
#define ANOTHER_KERNEL_PROFILER InstrumentKernelProfiler
#endif

////////////////////////////////////////////////// .... //////////////////////////////////////////////////

struct SessProfiler : public ISessProfiler {
  SessProfiler(const SessionState& sess_state,
               const ExecutionFrame& frame) : ISessProfiler(sess_state, frame)
#ifdef ANOTHER_SESS_PROFILER
                                              ,
                                              another_sess_profiler_(sess_state, frame)
#endif
  {
  }

#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESS_PROFILER another_sess_profiler_;
#endif
};

struct KernelProfiler : public IKernelProfiler {
  KernelProfiler(SessProfiler& sess_profiler,
                 const OpKernelContextInternal& context,
                 const onnxruntime::OpKernel& kernel) : IKernelProfiler(sess_profiler, context, kernel)
#ifdef ANOTHER_KERNEL_PROFILER
                                                        ,
                                                        another_kernel_profiler_(sess_profiler.GetAnotherSessProfiler(), context, kernel)
#endif
  {
  }
  ~KernelProfiler() {
  }
#ifdef ANOTHER_KERNEL_PROFILER
  ANOTHER_KERNEL_PROFILER another_kernel_profiler_;
#endif
};

}  // namespace onnxruntime-