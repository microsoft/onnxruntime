// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ORT_MEMORY_PROFILE

#include "session_profiler_base.h"
namespace onnxruntime {

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
  ISessProfiler& GetAnotherSessProfiler() override { return another_sess_profiler_; }
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
}  // namespace onnxruntime
#endif