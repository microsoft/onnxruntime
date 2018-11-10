// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(ExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger,
                                   const std::vector<const NodeArg*>& implicit_inputs)
      : OpKernelContext(&frame, &kernel, logger),
        implicit_inputs_{implicit_inputs} {
  }

  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return GetSessionState().GetSubgraphSessionState(GetNodeIndex(), attribute_name);
  }

  const MLValue* GetInputMLValue(int index) const {
    return OpKernelContext::GetInputMLValue(index);
  }

  MLValue* GetOutputMLValue(int index) {
    return OpKernelContext::GetOutputMLValue(index);
  }

  std::unordered_map<std::string, const MLValue*> GetImplicitInputs() const {
    // we need to convert implicit_inputs_ to a name to MLValue map so it can be used in the ExecutionFrame
    // for a subgraph (the index numbers will be different there).
    std::unordered_map<std::string, const MLValue*> implicit_inputs_map;

    for (int i = 0, end = gsl::narrow_cast<int>(implicit_inputs_.size()); i < end; ++i) {
      implicit_inputs_map[implicit_inputs_[i]->Name()] = GetImplicitInputMLValue(i);
    }

    return implicit_inputs_map;
  }

 private:
  const std::vector<const NodeArg*>& implicit_inputs_;
};

}  // namespace onnxruntime
