// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/session/onnxruntime_c_api.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;
class ExecutionFrame;

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(const SessionState& session_state,
                                   IExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger,
                                   const RunOptions& run_options)
      : OpKernelContext(&frame, &kernel, session_state.GetThreadPool(), logger),
        session_state_(session_state),
        run_options_(run_options) {
    const auto& implicit_inputs = kernel.Node().ImplicitInputDefs();
    int num_implicit_inputs = static_cast<int>(implicit_inputs.size());
    implicit_input_values_.reserve(num_implicit_inputs);

    for (int i = 0; i < num_implicit_inputs; ++i) {
      const auto* entry = GetImplicitInputMLValue(i);
      ORT_ENFORCE(entry != nullptr, "All implicit inputs should have OrtValue instances by now. ",
                  implicit_inputs[i]->Name(), " does not.");
      implicit_input_values_.push_back(entry);
    }
  }

  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return session_state_.GetSubgraphSessionState(GetNodeIndex(), attribute_name);
  }

  const OrtValue* GetInputMLValue(int index) const {
    return OpKernelContext::GetInputMLValue(index);
  }

  OrtValue* GetOutputMLValue(int index) {
    return OpKernelContext::GetOutputMLValue(index);
  }

  OrtValue* OutputMLValue(int index, const TensorShape& shape) {
    return OpKernelContext::OutputMLValue(index, shape);
  }

  // Get the OrtValue's for all implicit inputs. Order is same as Node::ImplicitInputDefs(). No nullptr entries.
  const std::vector<const OrtValue*>& GetImplicitInputs() const {
    return implicit_input_values_;
  }

  const RunOptions& GetRunOptions() const noexcept { return run_options_; }

 private:
  const SessionState& session_state_;
  const RunOptions& run_options_;
  std::vector<const OrtValue*> implicit_input_values_;
};

}  // namespace onnxruntime
