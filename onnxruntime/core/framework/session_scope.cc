// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "session_scope.h"
#include "core/common/profiler.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

using namespace onnxruntime::profiling;

#ifdef ORT_MINIMAL_BUILD

SessionScope::SessionScope(const SessionState&, const ExecutionFrame&) {}

SessionScope::~SessionScope() {}

KernelScope::KernelScope(SessionScope&, OpKernelContextInternal&, const OpKernel&) {}

KernelScope::~KernelScope() {}

#else

class IScope {
 public:
  IScope() = default;
  virtual ~IScope() = 0;
  //virtual void Start(){};
  //virtual void Stop(){};
};


#ifdef CONCURRENCY_VISUALIZER
#include <cvmarkersobj.h>
using namespace Concurrency;

class ConcurrencyKernelScope;
/// <summary>
/// ConcurrencyScope
/// </summary>
class ConcurrencyScope : public IScope {
 public:
  friend class ConcurrencyKernelScope;
  ConcurrencyScope(const GraphViewer& graph_viewer) : series_(ComposeSeriesName(graph_viewer)) {}
  ~ConcurrencyScope() = default;

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
  ConcurrencyKernelScope(ConcurrencyScope& scope,
                         const OpKernel& kernel) : span_(scope.series_,
                                                         "%s.%d",
                                                         kernel.Node().OpType().c_str(),
                                                         kernel.Node().Index()) {
    scope.series_.write_flag(kernel.Node().Name().c_str());
  };

  ~ConcurrencyKernelScope() = default;

 private:
  diagnostic::span span_;
};

#else

class ConcurrencyScope : public IScope {
 public:
  ConcurrencyScope(const GraphViewer&) {}
  ~ConcurrencyScope() = default;
};

class ConcurrencyKernelScope {
 public:
  ConcurrencyKernelScope(ConcurrencyScope&, const OpKernel&);
};

#endif

#ifdef ENABLE_NVTX_PROFILE

#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"

/// <summary>
/// NVTXScope
/// </summary>
class NVTXScope : public IScope {
 public:
  NVTXScope(const std::thread::id& thread_id) : sess_tag_(profile::Context::GetInstance().GetThreadTagOrDefaul(thread_id)),
                                           forward_range_("Batch-" + session_tag_ + " Forward", profile::Color::White),
                                           backward_range_("Batch-" + session_tag_ + " Backward", profile::Color::Black) {}
  ~NVTXScope() {
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
  NVTXKernelScope(const OpKernel& kernel) : node_compute_range_(MakeString(kernel.Node().OpType(), ".",
                                                                           kernel.Node().Index(), "(",
                                                                           kernel.Node().Name(), ")"),
                                                                profile::Color::Yellow) {
    node_compute_range_.Begin();
  }
  ~NVTXKernelScope() {
    NVTXKernelScope.End();
  }

 private:
  profile::NvtxRangeCreator node_compute_range_;
};

#else

class NVTXScope : public IScope {
 public:
  NVTXScope(const std::thread::id&) {}
  ~NVTXScope() = default;
};

class NVTXKernelScope {
 public:
  NVTXKernelScope(const OpKernel&);
};

#endif

/// <summary>
/// DumpInOutScope
/// </summary>
//class DumpInOutScope : public IScope {
// public:
//  DumpInOutScope(size_t) {}
//  ~DumpInOutScope() = default;
//};
#ifdef DEBUG_NODE_INPUTS_OUTPUTS

class DumpKernelScope {
 public:
  DumpKernelScope(const SessionState& sess_state;
                  OpKernelContextInternal & context,
                  const OpKernel& kernel,
                  size_t iteration) : sess_state_(sess_state),
                                      context_(context),
                                      kernel_(kernel),
                                      dump_ctx_(iteration, kernel_.Node().NodeIndex()) {
    utils::DumpNodeInputs(dump_ctx_, context_, kernel_.Node(), sess_state_);
  }

  DumpKernelScope() {
    utils::DumpNodeOutputs(dump_ctx_, context_, kernel_.Node(), sess_state_);
  }

 private:
  const SessionState& sess_state_;
  OpKernelContextInternal& context_;
  const OpKernel& kernel_;
  size_t iteration_;
  utils::NodeDumpContext dump_ctx_;
};

#else

class DumpKernelScope {
 public:
  DumpKernelScope(const SessionState&,
                  OpKernelContextInternal&,
                  const OpKernel&,
                  size_t) {}
};

#endif

/// <summary>
/// ProfilerScope
/// </summary>
class ProfilerScope : public IScope {
 public:
  ProfilerScope(Profiler& profiler) : profiler_(profiler) {
    enabled_ = profiler_.IsEnabled();
    if (enabled_) {
      sess_start_ = profiler_.Start();
    }
  }
  ~ProfilerScope() {
    if (enabled_) {
      profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "SequentialExecutor::Execute", sess_start_);
    }
  }

 private:
  bool enabled_;
  Profiler& profiler_;
  TimePoint sess_start_;
};

/// <summary>
/// SessionScopeImpl
/// </summary>
class SessionScopeImpl {
 public:
  friend class KernelScopeImpl;
  SessionScopeImpl(const SessionState& sess_state,
                   const ExecutionFrame& frame) : sess_state_(sess_state),
                                                  frame_(frame),
                                                  concurrency_scope_(sess_state.GetGraphViewer()),
                                                  profiler_scope_(sess_state.Profiler()),
                                                  nvtx_scope_(std::this_thread::get_id()) {
    iteration_++;
  };
  ~SessionScopeImpl() {}

 private:
  const SessionState& sess_state_;
  const ExecutionFrame& frame_;
  ConcurrencyScope concurrency_scope_;
  ProfilerScope profiler_scope_;
  NVTXScope nvtx_scope_;
  std::atomic<size_t> iteration_{0};
};

SessionScope::SessionScope(const SessionState& sess_state, const ExecutionFrame& frame) {
  impl_ = std::make_unique<SessionScopeImpl>(sess_state, frame);
}

class KernelScopeImpl {
 public:
  KernelScopeImpl(OpKernelContextInternal& context,
                  const OpKernel& kernel,
                  SessionScope& sess_scope) : context_(context),
                                              kernel_(kernel),
                                              sess_scope_(sess_scope),
                                              concur_scope_(sess_scope.impl_->concurrency_scope_, kernel),
                                              nvtx_scope_(kernel),
                                              dump_scope_(sess_scope.impl_->sess_state_, context, kernel, sess_scope.impl_->iteration_) {}
  ~KernelScopeImpl() = default;

 private:

  TimePoint kernel_begin_time_;
  std::string node_name_;
  OpKernelContextInternal& context_;
  const OpKernel& kernel_;
  SessionScope& sess_scope_;
  size_t input_activation_sizes_{};
  size_t input_parameter_sizes_{};
  size_t total_output_sizes_{};
  std::string input_type_shape_{};
  std::string output_type_shape_{};

  ConcurrencyKernelScope concur_scope_;
  NVTXKernelScope nvtx_scope_;
  DumpKernelScope dump_scope_;
};

KernelScope::KernelScope(OpKernelContextInternal& kernel_context,
                         const OpKernel& kernel, SessionScope& sess_scope) {
  impl_ = std::make_unique<KernelScopeImpl>(kernel_context, kernel, sess_scope);
}

#endif

}  // namespace onnxruntime