// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/fence.h"
#include "opencl_utils.h"

namespace onnxruntime {

/* Explanation on how it works and why is two events are needed:

~~~~~~> a flow of execution, a.k.a, a cpu thread or a cl_command_queue in time domain

consider two seperate flows of execution:

              [T as output]
             /
~~~~~~>(Kernel1)~~~~~~> queue 1

    [T as input]
        \
~~~~~~>(Kernel2)~~~~~~> queue 2

To make the execution is well-formed:

          [T] as output
          /
~~~~~~>(Kernel1)~~>(e1)~~~~~~~~~~~~~~~~~~~~~~~~~> queue 1

                                [T] as input
                                  \
~~~~~~~~~~~~~~~~~~~~~~~~~>(e2)~~>(Kernel2)~~~~~~> queue 2

The invariant time_of_execution_end(Kernel1) < time_of_execution_start(Kernel2)
must be enforced. And the event is related with the shared tensor T, so the
IFence is bounded to it.

*/

class OpenCLFence : public IFence {
 public:
  explicit OpenCLFence(const OpenCLExecutionProvider& exec);
  virtual ~OpenCLFence();
  virtual void BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) override;
  virtual void BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) override;
  virtual void AfterUsedAsInput(int queue_id) override;
  virtual void AfterUsedAsOutput(int queue_id) override;
  virtual bool CanRelease() override;

 private:
  cl_event produced_;
  cl_event consumed_;
  const OpenCLExecutionProvider* exec_;
};

}  // namespace onnxruntime
