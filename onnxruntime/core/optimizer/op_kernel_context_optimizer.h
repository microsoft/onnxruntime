// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_optimizer.h"

namespace onnxruntime {

class OpKernelContextOptimizer : public onnxruntime::OpKernelContext {
 public:
  OpKernelContextOptimizer(GraphOptimizer& optimizer,
                           const OpKernel& kernel,
                           const logging::Logger& logger) : OpKernelContext(&kernel, logger),
                                                            optimizer_(optimizer) {}

  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;
  Tensor* Output(int index, const TensorShape& shape) override;
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;
  const MLValue* GetInputMLValue(int index) const;
  MLValue* GetOutputMLValue(int index);

  // The APIs below are all for sync between different execution providers.
  // Since optimizer only leverage CPU execution provider, so no need to implement them.
  Fence_t InputFence(int index) const {
    return nullptr;
  }
  Fence_t ImplicitInputFence(int index) const {
    return nullptr;
  }
  Fence_t OutputFence(int index) const {
    return nullptr;
  }

  // The APIs below are only called in if/loop/scan kernels
  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return nullptr;
  }
  std::unordered_map<std::string, const MLValue*> GetImplicitInputs() const {
    return std::unordered_map<std::string, const MLValue*>();
  }
  const bool& GetTerminateFlag() const noexcept {
    return true;
  }

protected:
  Status GetOrCreateOutputMLValue(int index, MLValue*& p_value);

private:
  GraphOptimizer& optimizer_;
};

}  // namespace onnxruntime