// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
using namespace ::onnxruntime::common;
namespace onnxruntime {

std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) {
  return std::make_unique<OpKernelInfo>(info);
}

const onnxruntime::Node& OpKernel::Node() const {
  return op_kernel_info_->node();
}

const onnxruntime::KernelDef& OpKernel::KernelDef() const {
  return op_kernel_info_->GetKernelDef();
}

const OrtMemoryInfo& OpKernel::Allocator(int id, OrtMemType mem_type) const {
  return op_kernel_info_->GetMemoryInfo(id, mem_type);
}

OpKernelContext::OpKernelContext(_Inout_ IExecutionFrame* frame, _In_ const OpKernel* kernel,
                                 _In_opt_ concurrency::ThreadPool* threadpool, _In_ const logging::Logger& logger)
    : execution_frame_(frame), kernel_(kernel), threadpool_(threadpool), logger_(&logger) {
  ORT_ENFORCE(frame != nullptr, "Execution frame was null");
  ORT_ENFORCE(kernel != nullptr, "OpKernel was null");

  node_input_start_index_ = frame->GetNodeOffset(kernel->Node().Index());
  node_implicit_input_start_index_ = node_input_start_index_ + InputCount();
  node_output_start_index_ = node_implicit_input_start_index_ + ImplicitInputCount();
}

OpKernelContext::OpKernelContext(_In_opt_ concurrency::ThreadPool* threadpool,
                                 _In_ const logging::Logger& logger) : threadpool_(threadpool), logger_(&logger) {}

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

#if !defined(DISABLE_SPARSE_TENSORS)
SparseTensor* OpKernelContext::OutputSparse(int index, const TensorShape& shape) {
  auto p_ml_value = OutputMLValue(index, shape);
  return p_ml_value ? p_ml_value->GetMutable<SparseTensor>() : nullptr;
}
#endif

bool OpKernelContext::TryGetInferredInputShape(int index, TensorShape& shape) const {
  return execution_frame_->TryGetInferredShape(GetInputArgIndex(index), shape);
}

bool OpKernelContext::TryGetInferredOutputShape(int index, TensorShape& shape) const {
  return execution_frame_->TryGetInferredShape(GetOutputArgIndex(index), shape);
}

OrtValue* OpKernelContext::OutputMLValue(int index, const TensorShape& shape) {
  if (index < 0 || index >= OutputCount())
    return nullptr;

  //: Though we don't need to give 'ret' an initial value, GCC would generate a warning if we don't do that
  //"error: 'ret' may be used uninitialized in this function"
  //This warning only exists in Release build.
  //I believe it's a false alarm.

  OrtValue* p_ml_value = nullptr;
  Status status = execution_frame_->GetOrCreateNodeOutputMLValue(index, GetOutputArgIndex(index), &shape, p_ml_value, kernel_->Node());
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
  auto status = execution_frame_->GetOrCreateNodeOutputMLValue(index, output_arg_index, nullptr, value, kernel_->Node());
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

const std::string& OpKernelContext::GetNodeName() const {
  return kernel_->Node().Name();
}

const std::string& OpKernelContext::GetOpDomain() const {
  return kernel_->KernelDef().Domain();
}

const std::string& OpKernelContext::GetOpType() const {
  return kernel_->Node().OpType();
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

#ifdef ENABLE_TRAINING
Status OpKernelContext::SetOutputMLValue(int index, const OrtValue& ort_value) {
  if (index < 0 || index >= OutputCount()) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Index out of range. " + std::to_string(index) +
                      " was specified, but " + "range is [0, " + std::to_string(OutputCount()) + ")");
  }

  auto output_arg_index = GetOutputArgIndex(index);
  return execution_frame_->SetOutputMLValue(output_arg_index, ort_value);
}
#endif

// EagerKernelContext
EagerKernelContext::EagerKernelContext(_In_ const OrtValue* const* input_values, _In_ size_t input_count,
                                       _Inout_ OrtValue* const* output_values, _In_ size_t output_count,
                                       _In_ AllocatorPtr allocator, _In_ onnxruntime::concurrency::ThreadPool* threadpool,
                                       _In_ const logging::Logger& logger) : OpKernelContext(threadpool, logger),
                                                                             input_values_(input_values),
                                                                             input_count_(input_count),
                                                                             output_values_(output_values),
                                                                             output_count_(output_count),
                                                                             allocator_(allocator) {}

int EagerKernelContext::NumVariadicInputs(size_t arg_num) const {
  auto ort_value = input_values_[arg_num];
  if (ort_value->IsTensor()) {
    return static_cast<int>(ort_value->Get<Tensor>().Shape().Size());
  } else if (ort_value->IsTensorSequence()) {
    return static_cast<int>(ort_value->Get<TensorSeq>().Size());
  } else if (ort_value->IsSparseTensor()) {
    return static_cast<int>(ort_value->Get<SparseTensor>().Values().Shape().Size());
  } else {
    return 0;
  }
}

MLDataType EagerKernelContext::InputType(int index) const {
  if (index >= input_count_) {
    return nullptr;
  } else {
    return input_values_[index]->Type();
  }
}

MLDataType EagerKernelContext::OutputType(int index) const {
  if (index >= output_count_) {
    return nullptr;
  } else {
    return output_values_[index]->Type();
  }
}

const OrtValue* EagerKernelContext::GetInputMLValue(int index) const {
  if (index >= input_count_) {
    return nullptr;
  } else {
    return input_values_[index];
  }
}

OrtValue* EagerKernelContext::OutputMLValue(int index, const TensorShape& shape) {
  if (index >= output_count_) {
    return nullptr;
  }
  OrtValue& ort_value = *output_values_[index];
  if (!ort_value.IsAllocated()) {
    if (ort_value.IsTensor()) {
      Tensor::InitOrtValue(ort_value.Type(), shape, allocator_, ort_value);
    } else if (ort_value.IsTensorSequence()) {
      auto ml_type = ort_value.Type();
      auto element_type = ml_type->AsSequenceTensorType()->GetElementType();
      auto p_sequence = std::make_unique<TensorSeq>(element_type);
      auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
      ort_value.Init(p_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());
    } else if (ort_value.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
      auto ml_type = ort_value.Type();
      auto element_type = ml_type->AsSparseTensorType()->GetElementType();
      SparseTensor::InitOrtValue(element_type, shape, allocator_, ort_value);
#else
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Sparse tensor is not supported in this build");
#endif
    }
  }
  return &ort_value;
}

OrtValue* EagerKernelContext::GetOrCreateOutputMLValue(int index) {
  if (index >= output_count_) {
    return nullptr;
  } else {
    return output_values_[index];
  }
}

bool EagerKernelContext::TryGetInferredInputShape(int /*index*/, TensorShape& /*shape*/) const {
  return false;  // no shape inference in eager mode
}

bool EagerKernelContext::TryGetInferredOutputShape(int /*index*/, TensorShape& /*shape*/) const {
  return false;  // no shape inference in eager mode
}

int EagerKernelContext::InputCount() const {
  return static_cast<int>(input_count_);
}

int EagerKernelContext::ImplicitInputCount() const {
  return 0;
}

int EagerKernelContext::OutputCount() const {
  return static_cast<int>(output_count_);
}

Status EagerKernelContext::GetTempSpaceAllocator(AllocatorPtr* output) const {
  *output = allocator_;
  return Status::OK();
}

Fence_t EagerKernelContext::InputFence(int index) const {
  if (index >= input_count_) {
    return nullptr;
  } else {
    return input_values_[index]->Fence();
  }
}

Fence_t EagerKernelContext::ImplicitInputFence(int /*index*/) const {
  return nullptr;
}

Fence_t EagerKernelContext::OutputFence(int index) const {
  if (index >= output_count_) {
    return nullptr;
  } else {
    return output_values_[index]->Fence();
  }
}

int EagerKernelContext::GetDeviceId() const {
  return 0;
}

void* EagerKernelContext::GetComputeStream() const {
  return nullptr;
}

}  // namespace onnxruntime
