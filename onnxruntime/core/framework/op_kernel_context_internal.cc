// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

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

int OpKernelContextInternal::NumVariadicInputs(size_t arg_num) const {
  auto& arg_counts = kernel_->Node().InputArgCount();

  ORT_ENFORCE(arg_num < arg_counts.size(), "Invalid arg_num of ", arg_num, ". Num args is ", arg_counts.size());

  return arg_counts[arg_num];
}

Status OpKernelContextInternal::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = execution_frame_->GetAllocator(kernel_->Allocator(0, OrtMemTypeDefault));
  if (!*output)
    return Status(common::ONNXRUNTIME, common::FAIL, "TempSpace allocator not found");
  return Status::OK();
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

Status OpKernelContextInternal::GetOrCreateOutputMLValue(int index, MLValue*& p_value) {
  auto output_arg_index = GetOutputArgIndex(index);
  MLValueAllocationParameters parameters;
  ORT_ENFORCE(execution_frame_->GetOrCreateNodeOutputMLValue(output_arg_index, parameters, p_value).IsOK());
  return Status::OK();
}

int OpKernelContextInternal::GetInputArgIndex(int index) const {
  return node_input_start_index_ + index;
}

int OpKernelContextInternal::GetImplicitInputArgIndex(int index) const {
  return node_implicit_input_start_index_ + index;
}

int OpKernelContextInternal::GetOutputArgIndex(int index) const {
  return node_output_start_index_ + index;
}

onnxruntime::NodeIndex OpKernelContextInternal::GetNodeIndex() const {
  return kernel_->Node().Index();
}

const SessionState& OpKernelContextInternal::GetSessionState() const {
  return execution_frame_->SessionState();
}

const MLValue* OpKernelContextInternal::GetInputMLValue(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  int input_arg_index = GetInputArgIndex(index);
  return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
}

const MLValue* OpKernelContextInternal::GetImplicitInputMLValue(int index) const {
  if (index < 0 || index >= ImplicitInputCount())
    return nullptr;

  int input_arg_index = GetImplicitInputArgIndex(index);
  return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
}

MLValue* OpKernelContextInternal::GetOutputMLValue(int index) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  return execution_frame_->GetMutableNodeInputOrOutputMLValue(output_arg_index);
}

}  // namespace onnxruntime