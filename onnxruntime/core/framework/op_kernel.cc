// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
using namespace ::onnxruntime::common;
namespace onnxruntime {

OpKernelContext::OpKernelContext(_Inout_ IExecutionFrame* frame, _In_ const OpKernel* kernel,
                                 _In_opt_ concurrency::ThreadPool* threadpool, _In_ const logging::Logger& logger)
    : execution_frame_(frame), kernel_(kernel), threadpool_(threadpool), logger_(&logger) {
  ORT_ENFORCE(frame != nullptr, "Execution frame was null");
  ORT_ENFORCE(kernel != nullptr, "OpKernel was null");

  node_input_start_index_ = frame->GetNodeOffset(kernel->Node().Index());
  node_implicit_input_start_index_ = node_input_start_index_ + InputCount();
  node_output_start_index_ = node_implicit_input_start_index_ + ImplicitInputCount();
}

Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  auto p_ml_value = OutputMLValue(index, shape);
  return p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
}

Tensor* OpKernelContext::Output(int index, const std::vector<int64_t>& shape) {
  return Output(index, TensorShape(shape));
}

Tensor* OpKernelContext::Output(int index, const std::initializer_list<int64_t>& shape) {
  return Output(index, TensorShape(shape));
}

#if !defined(ORT_MINIMAL_BUILD)
SparseTensor* OpKernelContext::Output(int index, size_t nnz, const TensorShape& shape) {
  auto p_ml_value = OutputMLValue(index, shape, nnz);
  return p_ml_value ? p_ml_value->GetMutable<SparseTensor>() : nullptr;
}
#endif

bool OpKernelContext::TryGetInferredInputShape(int index, TensorShape& shape) const {
  return execution_frame_->TryGetInferredShape(GetInputArgIndex(index), shape);
}

bool OpKernelContext::TryGetInferredOutputShape(int index, TensorShape& shape) const {
  return execution_frame_->TryGetInferredShape(GetOutputArgIndex(index), shape);
}

OrtValue* OpKernelContext::OutputMLValue(int index, const TensorShape& shape, size_t nnz) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  //: Though we don't need to give 'ret' an initial value, GCC would generate a warning if we don't do that
  //"error: 'ret' may be used uninitialized in this function"
  //This warning only exists in Release build.
  //I believe it's a false alarm.

  OrtValue* p_ml_value = nullptr;
  Status status = execution_frame_->GetOrCreateNodeOutputMLValue(GetOutputArgIndex(index), &shape, p_ml_value, nnz);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return p_ml_value;
}

int OpKernelContext::NumVariadicInputs(size_t arg_num) const {
  auto& arg_counts = kernel_->Node().InputArgCount();

  ORT_ENFORCE(arg_num < arg_counts.size(), "Invalid arg_num of ", arg_num, ". Num args is ", arg_counts.size());

  return arg_counts[arg_num];
}

Status OpKernelContext::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = execution_frame_->GetAllocator(kernel_->Allocator(0, OrtMemTypeDefault));
  if (!*output)
    return Status(common::ONNXRUNTIME, common::FAIL, "TempSpace allocator not found");
  return Status::OK();
}

MLDataType OpKernelContext::InputType(int index) const {
  int input_arg_index = GetInputArgIndex(index);
  const OrtValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

MLDataType OpKernelContext::OutputType(int index) const {
  auto output_arg_index = GetOutputArgIndex(index);
  const OrtValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Type() : nullptr;
}

Fence_t OpKernelContext::InputFence(int index) const {
  if (index >= InputCount())
    return nullptr;

  int input_index = GetInputArgIndex(index);
  const OrtValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Fence_t OpKernelContext::ImplicitInputFence(int index) const {
  if (index >= ImplicitInputCount())
    return nullptr;

  int input_index = GetImplicitInputArgIndex(index);
  const OrtValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(input_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

Fence_t OpKernelContext::OutputFence(int index) const {
  if (index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  const OrtValue* p_ml_value = execution_frame_->GetNodeInputOrOutputMLValue(output_arg_index);
  return p_ml_value ? p_ml_value->Fence() : nullptr;
}

OrtValue* OpKernelContext::GetOrCreateOutputMLValue(int index) {
  auto output_arg_index = GetOutputArgIndex(index);
  OrtValue* value = nullptr;
  auto status = execution_frame_->GetOrCreateNodeOutputMLValue(output_arg_index, nullptr, value);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return value;
}

int OpKernelContext::GetInputArgIndex(int index) const {
  return node_input_start_index_ + index;
}

int OpKernelContext::GetImplicitInputArgIndex(int index) const {
  return node_implicit_input_start_index_ + index;
}

int OpKernelContext::GetOutputArgIndex(int index) const {
  return node_output_start_index_ + index;
}

onnxruntime::NodeIndex OpKernelContext::GetNodeIndex() const {
  return kernel_->Node().Index();
}

const std::string& OpKernelContext::GetOpDomain() const {
  return kernel_->KernelDef().Domain();
}

const OrtValue* OpKernelContext::GetInputMLValue(int index) const {
  if (index < 0 || index >= InputCount())
    return nullptr;

  int input_arg_index = GetInputArgIndex(index);
  return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
}

const OrtValue* OpKernelContext::GetImplicitInputMLValue(int index) const {
  if (index < 0 || index >= ImplicitInputCount())
    return nullptr;

  int input_arg_index = GetImplicitInputArgIndex(index);
  return execution_frame_->GetNodeInputOrOutputMLValue(input_arg_index);
}

OrtValue* OpKernelContext::GetOutputMLValue(int index) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  auto output_arg_index = GetOutputArgIndex(index);
  return execution_frame_->GetMutableNodeInputOrOutputMLValue(output_arg_index);
}

}  // namespace onnxruntime
