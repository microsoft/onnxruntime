// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/torch/torch_custom_function_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include <thread>

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

void PythonOp::AddIntScalarArgs() {
  ORT_ENFORCE(const_args_.size() == const_arg_positions_.size());
  for (size_t i = 0; i < input_int_scalars_.size(); ++i) {
    const_arg_positions_.emplace_back(input_int_scalar_positions_.at(i));
    const_args_.emplace_back(Py_BuildValue("L", static_cast<long long>(input_int_scalars_.at(i))));
  }

  for (size_t i = 0; i < input_float_scalars_.size(); ++i) {
    const_arg_positions_.emplace_back(input_float_scalar_positions_.at(i));
    const_args_.emplace_back(Py_BuildValue("f", input_float_scalars_.at(i)));
  }
}

void PythonOp::AddInputTupleArgs() {
  ORT_ENFORCE(const_args_.size() == const_arg_positions_.size());
  for (size_t i = 0; i < input_int_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_int_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end = (i + 1 == input_int_tuple_begins_.size()) ? input_int_tuples_.size() : input_int_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("L", input_int_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }
    const_arg_positions_.emplace_back(input_int_tuple_positions_.at(i));
    const_args_.emplace_back(tuple);
  }
}

void PythonOp::AddFloatTupleArgs() {
  ORT_ENFORCE(const_args_.size() == const_arg_positions_.size());
  for (size_t i = 0; i < input_float_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_float_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end = (i + 1 == input_float_tuple_begins_.size()) ? input_float_tuples_.size() : input_float_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("f", input_float_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }
    const_arg_positions_.emplace_back(input_float_tuple_positions_.at(i));
    const_args_.emplace_back(tuple);
  }
}

void PythonOp::AddPointerScalarArgs() {
  ORT_ENFORCE(const_args_.size() == const_arg_positions_.size());
  for (size_t i = 0; i < input_pointer_scalars_.size(); ++i) {
    const_arg_positions_.emplace_back(input_pointer_scalar_positions_.at(i));
    PyObject* ptr = reinterpret_cast<PyObject*>(input_pointer_scalars_.at(i));
    const_args_.emplace_back(ptr);
  }
}

void PythonOp::CreateConstArgs() {
  ORT_ENFORCE(const_args_.size() == 0);
  ORT_ENFORCE(const_arg_positions_.size() == 0);
  PythonOp::AddIntScalarArgs();
  PythonOp::AddInputTupleArgs();
  PythonOp::AddFloatTupleArgs();
  PythonOp::AddPointerScalarArgs();
}

void PythonOp::CreateArgPositions() {
  ORT_ENFORCE(arg_positions_.size() == 0);

  // occupied[i] being true means the i-th input argument
  // to Python function has been set.
  std::vector<bool> occupied(input_tensor_types_.size() + const_args_.size(), false);

  // We know all non-tensors were set above, so let's catch up.
  for (const auto pos : const_arg_positions_) {
    occupied.at(pos) = true;
  }

  // Search for empty slots for tensors.
  // The i-th empty slot is assigned the i-th input tensor.
  for (size_t i = 0; i < occupied.size(); ++i) {
    if (occupied.at(i)) {
      continue;
    }
    // arg_positions[i] is the position index for the i-th tensor input.
    // Find an empty slot whose position index is i.
    arg_positions_.push_back(i);
  }
}

Status PythonOp::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  ORT_ENFORCE(nullptr != context);
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  auto outputs_count = (size_t)ctx_internal->OutputCount();

  std::vector<OrtValue*> inputs;
  std::vector<void*> outputs;
  for (size_t i = 0; i < inputs_count; ++i) {
    inputs.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(i)));
  }

  // Invoke python calls.
  std::string err;
  auto state = onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().GetGil();
  void* callback = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetForwardCore(name_);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Forward(
      callback, input_tensor_requires_grads_, inputs, arg_positions_, const_args_, const_arg_positions_, outputs);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().PutGil(state);

  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // Handle the outputs;
  // The 1st output is context index of auto grad function.
  // Other outputs are address of OrtValue we got from python script run.
  PyObject* ctx_addr = reinterpret_cast<PyObject*>(outputs[0]);
  ORT_ENFORCE(ctx_addr, "Context object pointer should not be null");
  int64_t ctx_index = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().RegisterContext(ctx_addr);

  // todo(pengwa): optional? re-visit this once we have better understanding on how PyTorch handle the ref cnt.
  Py_INCREF(ctx_addr);

  Tensor* first_output_tensor = context->Output(0, {1});
  ORT_ENFORCE(first_output_tensor != nullptr, "first output tensor should not be null.");
  int64_t* step_data_new = first_output_tensor->template MutableData<int64_t>();
  *step_data_new = ctx_index;

  for (size_t index = 1; index < outputs_count; ++index) {
    void* forward_ret_ortvalue_addr = outputs.at(index);
    auto* forward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(forward_ret_ortvalue_addr);
    ORT_ENFORCE(forward_ret_ortvalue_ptr != nullptr, "forward_ret_ortvalue_ptr should not be null");

    // Set the ORTValue directly as outputs.
    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(index, *forward_ret_ortvalue_ptr));
  }

  return Status::OK();
}

Status PythonOpGrad::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  ORT_ENFORCE(nullptr != context);
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  auto outputs_count = (size_t)ctx_internal->OutputCount();

  std::vector<OrtValue*> inputs;
  std::vector<void*> outputs;
  const Tensor* first_output_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(first_output_tensor != nullptr, "first output tensor should not be null.");
  const int64_t* context_index_ptr = first_output_tensor->template Data<int64_t>();
  PyObject* ctx_ptr = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetContext(*context_index_ptr);

  for (size_t i = 1; i < inputs_count; ++i) {
    inputs.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(i)));
  }

  std::vector<int64_t> arg_positions(inputs.size());
  for (int64_t i = 0; i < (int64_t)arg_positions.size(); ++i) {
    arg_positions.at(i) = i + 1;
  }

  std::vector<void*> const_args{ctx_ptr};
  std::vector<int64_t> const_arg_positions{0};
  std::string err;
  auto state = onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().GetGil();
  void* callback = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetBackwardCore(name_);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Backward(callback, input_tensor_requires_grads_, inputs, arg_positions, const_args, const_arg_positions, outputs);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().PutGil(state);
  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  outputs_count = outputs_count > outputs.size() ? outputs.size() : outputs_count;
  for (size_t index = 0; index < outputs_count; ++index) {
    void* backward_ret_ortvalue_addr = outputs[index];
    auto* backward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(backward_ret_ortvalue_addr);
    ORT_ENFORCE(backward_ret_ortvalue_ptr != nullptr, index, "th output from Python should not be null.");

    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(index, *backward_ret_ortvalue_ptr));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
