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
  explicit OpKernelContextInternal(ExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger,
                                   const bool& terminate_flag)
      : OpKernelContext(&kernel, logger),
        execution_frame_(&frame),
        terminate_flag_{terminate_flag} {

    node_input_start_index_ = frame.GetFirstArgIndex(kernel_->Node().Index());
    node_implicit_input_start_index_ = node_input_start_index_ + InputCount();
    node_output_start_index_ = node_implicit_input_start_index_ + ImplicitInputCount();
  }

  Tensor* Output(int index, const TensorShape& shape) override;
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;
  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;

  virtual Fence_t InputFence(int index) const;
  Fence_t ImplicitInputFence(int index) const;
  Fence_t OutputFence(int index) const;

  const SessionState* SubgraphSessionState(const std::string& attribute_name);
  std::unordered_map<std::string, const MLValue*> GetImplicitInputs() const;

  const MLValue* GetInputMLValue(int index) const;
  MLValue* GetOutputMLValue(int index);

  const bool& GetTerminateFlag() const noexcept { return terminate_flag_; }

protected:
  Status GetOrCreateOutputMLValue(int index, MLValue*& p_value) {
    auto output_arg_index = GetOutputArgIndex(index);
    MLValueAllocationParameters parameters;
    ORT_ENFORCE(execution_frame_->GetOrCreateNodeOutputMLValue(output_arg_index, parameters, p_value).IsOK());
    return Status::OK();
  }

private:
  onnxruntime::NodeIndex GetNodeIndex() const {
    return kernel_->Node().Index();
  }

  const SessionState& GetSessionState() const {
    return execution_frame_->SessionState();
  }

  int GetInputArgIndex(int index) const {
    return node_input_start_index_ + index;
  }

  int GetImplicitInputArgIndex(int index) const {
    return node_implicit_input_start_index_ + index;
  }

  int GetOutputArgIndex(int index) const {
    return node_output_start_index_ + index;
  }

  const MLValue* OpKernelContextInternal::GetImplicitInputMLValue(int index) const {
    if (index < 0 || index >= ImplicitInputCount())
      return nullptr;

    int input_arg_index = GetImplicitInputArgIndex(index);
    return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
  }

  ExecutionFrame* execution_frame_{nullptr};
  const bool& terminate_flag_;

  // The argument starting index in ExecutionFrame.
  int node_input_start_index_{-1};
  int node_implicit_input_start_index_{-1};
  int node_output_start_index_{-1};
};

}  // namespace onnxruntime
