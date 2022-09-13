// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

class SessionState;
class ExecutionFrame;

namespace onnxruntime {

struct ISessProfiler {
 protected:
  ISessProfiler(const SessionState& sess_state,
                const ExecutionFrame& frame) : sess_state_(sess_state),
                                               frame_(frame) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ISessProfiler);

 public:
  virtual ISessProfiler& GetAnotherSessProfiler() {
    ORT_ENFORCE(false, "must return an upstream profiler");
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
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IKernelProfiler);

  ISessProfiler& sess_profiler_;
  const OpKernelContextInternal& context_;
  const OpKernel& kernel_;
};

}  // namespace onnxruntime