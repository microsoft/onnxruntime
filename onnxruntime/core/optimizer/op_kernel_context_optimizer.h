// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class OpKernelContextOptimizer : public onnxruntime::OpKernelContext {
 public:
  OpKernelContextOptimizer(const OpKernel* kernel,
                           const logging::Logger& logger) : OpKernelContext(kernel, logger) {}

  int NumVariadicInputs(size_t arg_num) const;

  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;

  Tensor* Output(int index, const TensorShape& shape) override;
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;

  virtual Fence_t InputFence(int index) const = 0;

  Status GetOrCreateOutputMLValue(int index, MLValue*& value);

  onnxruntime::NodeIndex GetNodeIndex() const;
  const SessionState& GetSessionState() const;

  int GetInputArgIndex(int index) const;
  int GetImplicitInputArgIndex(int index) const;
  int GetOutputArgIndex(int index) const;
};

}  // namespace onnxruntime