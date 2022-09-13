// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT

#include <Windows.h>
#include "core/platform/tracing.h"
#include "session_profiler_base.h"

namespace onnxruntime {

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
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
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
}  // namespace onnxruntime
#endif
