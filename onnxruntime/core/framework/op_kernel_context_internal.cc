// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

OpKernelContextInternal::OpKernelContextInternal(ExecutionFrame& frame,
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

MLDataType OpKernelContextInternal::InputType(int index) const {
  int input_arg_index = GetInputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

MLDataType OpKernelContextInternal::OutputType(int index) const {
  auto output_arg_index = GetOutputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

Tensor* OpKernelContextInternal::Output(int index, const TensorShape& shape) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  // In this case, it's assumed that the tensor hasn't been allocated yet,
  // so that it's calling ExecutionFrame to create a tensor in the given position with given shape.
  MLValueAllocationParameters parameters{&shape};

  //: Though we don't need to give 'ret' an initial value, GCC would generate a warning if we don't do that
  //"error: 'ret' may be used uninitialized in this function"
  //This warning only exists in Release build.
  //I believe it's a false alarm.
  MLValue* p_ml_value = nullptr;
  Status status = execution_frame_->GetOrCreateNodeOutputMLValue(GetOutputArgIndex(index), parameters, p_ml_value);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
}

Status OpKernelContextInternal::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = execution_frame_->GetAllocator(kernel_->Allocator(0, OrtMemTypeDefault));
  if (!*output)
    return Status(common::ONNXRUNTIME, common::FAIL, "TempSpace allocator not found");
  return Status::OK();
}

Fence_t OpKernelContextInternal::InputFence(int index) const {
  if (index >= InputCount())
    return nullptr;

  int input_index = GetInputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Fence_t OpKernelContextInternal::ImplicitInputFence(int index) const {
  if (index >= ImplicitInputCount())
    return nullptr;

  int input_index = GetImplicitInputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Fence_t OpKernelContextInternal::OutputFence(int index) const {
  if (index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  const MLValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

const SessionState* OpKernelContextInternal::SubgraphSessionState(const std::string& attribute_name) {
  return GetSessionState().GetSubgraphSessionState(GetNodeIndex(), attribute_name);
}

std::unordered_map<std::string, const MLValue*> OpKernelContextInternal::GetImplicitInputs() const {
  // we need to convert implicit_inputs_ to a name to MLValue map so it can be used in the ExecutionFrame
  // for a subgraph (the index numbers will be different there).
  std::unordered_map<std::string, const MLValue*> implicit_inputs_map;
  const std::vector<NodeArg*>& implicit_inputs = kernel_->Node().ImplicitInputDefs();

  for (int i = 0, end = gsl::narrow_cast<int>(implicit_inputs.size()); i < end; ++i) {
    implicit_inputs_map[implicit_inputs[i]->Name()] = GetImplicitInputMLValue(i);
  }

  return implicit_inputs_map;
}

const MLValue* OpKernelContextInternal::GetInputMLValue(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  int input_arg_index = GetInputArgIndex(index);
  return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
}

MLValue* OpKernelContextInternal::GetOutputMLValue(int index) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  return execution_frame_->GetMutableNodeInputOrOutputMLValue(output_arg_index);
}

}  // namespace onnxruntime