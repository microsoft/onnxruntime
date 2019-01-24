// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/execution_frame.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(ExecutionFrame* frame,
                                   const OpKernel* kernel,
                                   const logging::Logger& logger,
                                   const std::vector<NodeArg*>& implicit_inputs,
                                   const bool& terminate_flag)
      : OpKernelContext(kernel, logger),
        execution_frame_(frame),
        implicit_inputs_{implicit_inputs},
        terminate_flag_{terminate_flag} {
    ORT_ENFORCE(frame != nullptr, "Execution frame was null");

    node_input_start_index_ = frame->GetFirstArgIndex(kernel_->Node().Index());
    node_implicit_input_start_index_ = node_input_start_index_ + InputCount();
    node_output_start_index_ = node_implicit_input_start_index_ + ImplicitInputCount();
  }

  int NumVariadicInputs(size_t arg_num) const;

  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;

  Tensor* Output(int index, const TensorShape& shape) override;
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;

  virtual Fence_t InputFence(int index) const = 0;

  /**
  Return the fence of current node's implicit input.
  @param index The index of the implicit input.
  @returns Point to the Fence of the implicit input MLValue.
  It is null if the input MLValue doesn't have fence or the input is optional.
  */
  virtual Fence_t ImplicitInputFence(int index) const = 0;

  /**
  Return the fence of current node's output identifed by index.
  @param index The index of the output.
  @returns Point to the Fence of the output MLValue.
  It is null if the output MLValue doesn't have fence or the output is optional.
  */
  virtual Fence_t OutputFence(int index) const = 0;


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

  const bool& GetTerminateFlag() const noexcept { return terminate_flag_; }

 private:
  ExecutionFrame* execution_frame_{nullptr};
  const std::vector<NodeArg*>& implicit_inputs_;
  const bool& terminate_flag_;
};

}  // namespace onnxruntime
