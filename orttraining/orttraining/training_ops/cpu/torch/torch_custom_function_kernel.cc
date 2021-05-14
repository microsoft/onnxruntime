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

PythonOp::PythonOp(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
  inplace_ = info.GetAttrOrDefault("inplace", static_cast<int64_t>(0));
  is_training_mode_ = static_cast<bool>(info.GetAttrOrDefault("training_mode", static_cast<int64_t>(0)));
  ORT_THROW_IF_ERROR(info.GetAttr("call_convention", &call_convention_));

  // Input tensors.
  input_tensor_types_ = info.GetAttrsOrDefault("input_tensor_types", std::vector<int64_t>());
  input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());

  ORT_ENFORCE(input_tensor_types_.size() == Node().InputDefs().size());

  // Input int scalars.
  input_int_scalars_ = info.GetAttrsOrDefault("input_int_scalars", std::vector<int64_t>());
  input_int_scalar_positions_ = info.GetAttrsOrDefault("input_int_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_int_scalars_.size() == input_int_scalar_positions_.size());

  // Input float scalars.
  input_float_scalars_ = info.GetAttrsOrDefault("input_float_scalars", std::vector<float>());
  input_float_scalar_positions_ = info.GetAttrsOrDefault("input_float_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_float_scalars_.size() == input_float_scalar_positions_.size());

  // Input int tuples.
  input_int_tuples_ = info.GetAttrsOrDefault("input_int_tuples", std::vector<int64_t>());
  input_int_tuple_positions_ = info.GetAttrsOrDefault("input_int_tuple_positions", std::vector<int64_t>());
  input_int_tuple_begins_ = info.GetAttrsOrDefault("input_int_tuple_begins", std::vector<int64_t>());

  ORT_ENFORCE(input_int_tuple_positions_.size() == input_int_tuple_begins_.size());

  // Input float tuples.
  input_float_tuples_ = info.GetAttrsOrDefault("input_float_tuples", std::vector<float>());
  input_float_tuple_positions_ = info.GetAttrsOrDefault("input_float_tuple_positions", std::vector<int64_t>());
  input_float_tuple_begins_ = info.GetAttrsOrDefault("input_float_tuple_begins", std::vector<int64_t>());

  ORT_ENFORCE(input_float_tuple_positions_.size() == input_float_tuple_begins_.size());

  input_pointer_scalars_ = info.GetAttrsOrDefault("input_pointer_scalars", std::vector<int64_t>());
  input_pointer_scalar_positions_ = info.GetAttrsOrDefault("input_pointer_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_pointer_scalars_.size() == input_pointer_scalar_positions_.size());

  // Output tensors.
  output_tensor_types_ = info.GetAttrsOrDefault("output_tensor_types", std::vector<int64_t>());
  output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());

  CreateConstArgs();
  CreateArgPositions();
}

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
  AddIntScalarArgs();
  AddInputTupleArgs();
  AddFloatTupleArgs();
  AddPointerScalarArgs();
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

std::vector<OrtValue*> CreateArgs(OpKernelContext* context, const size_t begin_index) {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  std::vector<OrtValue*> args;
  for (size_t i = begin_index; i < static_cast<size_t>(ctx_internal->InputCount()); ++i) {
    args.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(i)));
  }
  return args;
}

void SetContextOutput(OpKernelContext* context, std::vector<void*>& returned_args) {
  // Handle the outputs;
  // The 1st output is context index of auto grad function.
  // Other outputs are address of OrtValue we got from python script run.
  PyObject* ctx_addr = reinterpret_cast<PyObject*>(returned_args[0]);
  ORT_ENFORCE(ctx_addr, "Context object pointer should not be null");

  // todo(pengwa): optional? re-visit this once we have better understanding on how PyTorch handle the ref cnt.
  Py_INCREF(ctx_addr);

  Tensor* ctx_id_tensor = context->Output(0, {1});
  ORT_ENFORCE(ctx_id_tensor != nullptr, "Context tensor should not be null.");
  int64_t* ctx_id_data = ctx_id_tensor->template MutableData<int64_t>();
  *ctx_id_data = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().RegisterContext(ctx_addr);
}

void SetOtherOutputs(OpKernelContext* context, std::vector<void*>& returned_args) {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  for (size_t i = 1; i < static_cast<size_t>(ctx_internal->OutputCount()); ++i) {
    // The returned pointer points to a OrtValue created on Python side.
    // Here we just cast it to the right type.
    OrtValue* ptr = reinterpret_cast<OrtValue*>(returned_args.at(i));
    ORT_ENFORCE(ptr, "Returned argument from Python should not be null.");
    // Set the output directly to Python-generated OrtValue value.
    ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(i, *ptr));
  }
}

Status PythonOp::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // Create non-constant arguments for calling Python function.
  // Constant arguments are created in ctor.
  std::vector<OrtValue*> args = CreateArgs(context, 0);
  // Place holder for Python returned values.
  std::vector<void*> returned_args;

  // Invoke python calls.
  std::string err;
  void* callback = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetForwardCore(name_);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Forward(
      callback, input_tensor_requires_grads_, args, arg_positions_, const_args_, const_arg_positions_, returned_args, is_training_mode_);

  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  // First output of this op is Pytorch autograd's context.
  SetContextOutput(context, returned_args);
  // Other outputs are wrappers of Pytorch tensors.
  SetOtherOutputs(context, returned_args);
  return Status::OK();
}

void PythonOpGrad::SetPositions() {
  ORT_ENFORCE(const_arg_positions_.size() == 0);
  ORT_ENFORCE(arg_positions_.size() == 0);

  // Pytorch's autograd context is the first (indexed by 0) input of the called Python function.
  // Note that here we will call autograd.Function.backward(ctx, tensor0, tensor1, ...).
  const_arg_positions_ = {0};

  // The rest inputs are just Pytorch tensors.
  arg_positions_.resize(input_tensor_types_.size());
  for (size_t i = 0; i < arg_positions_.size(); ++i) {
    // i-th tensor is the (i+1)-th input of autograd.Function.backward.
    arg_positions_.at(i) = static_cast<int64_t>(i) + 1;
  }
}

std::vector<void*> CreateConstArgs(OpKernelContext* context) {
  const Tensor* context_id_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(context_id_tensor, "Context ID (first input) should not be null.");
  const int64_t* context_index_ptr = context_id_tensor->template Data<int64_t>();
  void* ctx_ptr = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetContext(*context_index_ptr);
  return {ctx_ptr};
}

void SetOutputs(OpKernelContext* context, std::vector<void*>& returned_args) {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  auto outputs_count = static_cast<size_t>(ctx_internal->OutputCount());
  // It's possible that Pytorch returns None as gradient and ORT Python side may skip them.
  // In that case, returned_args may contain less arguments.
  outputs_count = outputs_count > returned_args.size() ? returned_args.size() : outputs_count;
  for (size_t i = 0; i < outputs_count; ++i) {
    OrtValue* ptr = reinterpret_cast<OrtValue*>(returned_args.at(i));
    ORT_ENFORCE(ptr, i, "th output from Python should not be null.");
    ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(i, *ptr));
  }
}

PythonOpGrad::PythonOpGrad(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
  ORT_THROW_IF_ERROR(info.GetAttrs("input_tensor_types", input_tensor_types_));
  ORT_THROW_IF_ERROR(info.GetAttrs("output_tensor_types", output_tensor_types_));
  input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());
  output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());
  SetPositions();
}

Status PythonOpGrad::ComputeInternal(OpKernelContext* context) const {
  // Todo(pengwa): perf impact and how much, leave it now to guarantee correctness.
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  auto args = CreateArgs(context, 1);
  // This is called "const" because that's how Pytorch calls all non-tensor inputs.
  auto const_args = CreateConstArgs(context);
  std::vector<void*> returned_args;

  std::string err;
  void* callback = onnxruntime::language_interop_ops::torch::OrtTorchFunctionPool::GetInstance().GetBackwardCore(name_);
  onnxruntime::language_interop_ops::torch::TorchProxy::GetInstance().Backward(
      callback, input_tensor_requires_grads_, args, arg_positions_, const_args, const_arg_positions_, returned_args);
  // todo(pengwa): okay to remove it?
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  SetOutputs(context, returned_args);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
