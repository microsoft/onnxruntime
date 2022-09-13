// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef CONCURRENCY_VISUALIZER

#include "session_profiler_base.h"
#include <cvmarkersobj.h>

namespace onnxruntime {

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
                                                      series_(static_cast<ConcurrencySessProfiler&>(sess_profiler).series_),
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
}  // namespace onnxruntime

#endif