// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "session_profiler_base.h"
#include "concurrency_profiler.h"
#include "nvtx_profiler.h"
#include "dump_profiler.h"
#include "mem_profiler.h"
#include "perf_profiler.h"

namespace onnxruntime {

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
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
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

}  // namespace onnxruntime