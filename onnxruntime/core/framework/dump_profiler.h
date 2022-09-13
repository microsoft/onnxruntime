// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

#include "session_profiler_base.h"
#include "core/framework/debug_node_inputs_outputs_utils.h"

namespace onnxruntime {

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
}  // namespace onnxruntime
#endif