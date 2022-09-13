// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_NVTX_PROFILE

#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#include "session_profiler_base.h"

namespace onnxruntime {

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
                                               sess_profiler_(static_cast<NVTXSessProfiler&>(sess_profiler)),
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
}  // namespace onnxruntime
#endif