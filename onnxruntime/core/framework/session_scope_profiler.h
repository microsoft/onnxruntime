// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

//#include <memory>
//#include <string>

#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
#endif
#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif

namespace onnxruntime {

struct ISessProfiler {
 protected:
  ISessProfiler(const SessionState& sess_state,
                const ExecutionFrame& frame) : sess_state_(sess_state),
                                               frame_(frame) {}
  const SessionState& sess_state_;
  const ExecutionFrame& frame_;

 public:
  virtual void ProfileKernelBegin(const OpKernelContextInternal&, const onnxruntime::OpKernel&){}
  virtual void ProfileKernelEnd(const OpKernelContextInternal&, const onnxruntime::OpKernel&){}
};

///////////////////////////////////////////////// CONCURRENCY /////////////////////////////////////////////////
#ifdef CONCURRENCY_VISUALIZER
using namespace Concurrency;
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

  void ProfileKernelBegin(const OpKernelContextInternal&, const onnxruntime::OpKernel&) override {
  
  }

  void ProfileKernelEnd(const OpKernelContextInternal&, const onnxruntime::OpKernel&) override {
  
  }

  diagnostic::marker_series series_;
  diagnostic::span span_;

#ifdef ANOTHER_SESS_PROFILER
  ANOTHER_SESSION_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER ConcurrencySessProfiler
#endif

////////////////////////////////////////////////////// NVTX //////////////////////////////////////////////////////
#ifdef ENABLE_NVTX_PROFILE
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
  ANOTHER_SESSION_PROFILER another_sess_profiler_;
#endif
};
#undef ANOTHER_SESS_PROFILER
#define ANOTHER_SESS_PROFILER NVTXSessProfiler
#endif

///////////////////////////////////////// DUMP INPUT AND OUTPUT ///////////////////////////////////////////

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#endif

//////////////////////////////////////////// MEM ////////////////////////////////////////////

#ifdef ORT_MEMORY_PROFILE
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
#endif

////////////////////////////////////////////////// PERF //////////////////////////////////////////////////

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

////////////////////////////////////////////////// TRACE //////////////////////////////////////////////////

#ifdef TRACE_EXECUTION
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
#endif

////////////////////////////////////////////////// .... //////////////////////////////////////////////////

struct SessProfiler : public ISessProfiler {
  SessProfiler(const SessionState& sess_state,
               const ExecutionFrame& frame) : ISessProfiler(sess_state, frame), another_sess_profiler_(sess_state, frame) {
  }

  void ProfileKernelBegin(const OpKernelContextInternal& context, const onnxruntime::OpKernel& kernel) override {
    another_sess_profiler_.ProfileKernelBegin(context, kernel);
  }

  void ProfileKernelEnd(const OpKernelContextInternal& context, const onnxruntime::OpKernel& kernel) override {
    another_sess_profiler_.ProfileKernelEnd(context, kernel);
  }

  ANOTHER_SESS_PROFILER another_sess_profiler_;
};

struct KernelProfiler {
  KernelProfiler(SessProfiler& sess_profiler,
                 const OpKernelContextInternal& context,
                 const onnxruntime::OpKernel& kernel) : sess_profiler_(sess_profiler),
                                                        context_(context),
                                                        kernel_(kernel) {
    sess_profiler_.ProfileKernelBegin(context_, kernel_);
  }
  ~KernelProfiler() {
    sess_profiler_.ProfileKernelEnd(context_, kernel_);
  }
  SessProfiler& sess_profiler_;
  const OpKernelContextInternal& context_;
  const onnxruntime::OpKernel& kernel_;
};

}  // namespace onnxruntime