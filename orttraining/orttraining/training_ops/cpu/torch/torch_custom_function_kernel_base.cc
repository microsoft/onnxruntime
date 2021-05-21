// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#include "core/framework/op_kernel_context_internal.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"

using namespace onnxruntime::language_interop_ops::torch;

namespace onnxruntime {
namespace contrib {

std::vector<OrtValue*> CreateOrtValueArgs(OpKernelContext* context,
                                          const size_t begin_index,
                                          const size_t num_arg) {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  std::vector<OrtValue*> args;
  for (size_t i = 0; i < num_arg; ++i) {
    args.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(begin_index + i)));
  }
  return args;
}

PythonOpBase::PythonOpBase(const OpKernelInfo& info) {
  ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
  inplace_ = info.GetAttrOrDefault("inplace", static_cast<int64_t>(0));
  is_training_mode_ = static_cast<bool>(info.GetAttrOrDefault("training_mode", static_cast<int64_t>(0)));
  ORT_THROW_IF_ERROR(info.GetAttr("call_convention", &call_convention_));

  // Input tensors.
  input_tensor_types_ = info.GetAttrsOrDefault("input_tensor_types", std::vector<int64_t>());
  input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());

  ORT_ENFORCE(input_tensor_types_.size() == info.node().InputDefs().size());

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

void PythonOpBase::AddIntScalarArgs() {
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

void PythonOpBase::AddInputTupleArgs() {
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

void PythonOpBase::AddFloatTupleArgs() {
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

void PythonOpBase::AddPointerScalarArgs() {
  ORT_ENFORCE(const_args_.size() == const_arg_positions_.size());
  for (size_t i = 0; i < input_pointer_scalars_.size(); ++i) {
    const_arg_positions_.emplace_back(input_pointer_scalar_positions_.at(i));
    PyObject* ptr = reinterpret_cast<PyObject*>(input_pointer_scalars_.at(i));
    const_args_.emplace_back(ptr);
  }
}

void PythonOpBase::CreateConstArgs() {
  ORT_ENFORCE(const_args_.size() == 0);
  ORT_ENFORCE(const_arg_positions_.size() == 0);
  AddIntScalarArgs();
  AddInputTupleArgs();
  AddFloatTupleArgs();
  AddPointerScalarArgs();
}

void PythonOpBase::CreateArgPositions() {
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

void PythonOpBase::SetContextOutput(OpKernelContext* context, void* diff_ctx) const {
  PyObject* ctx_addr = reinterpret_cast<PyObject*>(diff_ctx);
  ORT_ENFORCE(ctx_addr, "Context object pointer should not be null");

  Tensor* ctx_id_tensor = context->Output(0, {1});
  ORT_ENFORCE(ctx_id_tensor != nullptr, "Context tensor should not be null.");
  int64_t* ctx_id_data = ctx_id_tensor->template MutableData<int64_t>();
  *ctx_id_data = OrtTorchFunctionPool::GetInstance().RegisterContext(ctx_addr);
}

void PythonOpBase::SetOtherOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) const {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  for (size_t i = 0; i < returned_ortvalues.size(); ++i) {
    ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(i + 1, returned_ortvalues[i]));
  }
}

PythonOpGradBase::PythonOpGradBase(const OpKernelInfo& info) {
  ORT_THROW_IF_ERROR(info.GetAttr("name", &name_));
  ORT_THROW_IF_ERROR(info.GetAttr("inplace", &inplace_));
  ORT_THROW_IF_ERROR(info.GetAttrs("input_tensor_types", input_tensor_types_));
  ORT_THROW_IF_ERROR(info.GetAttrs("output_tensor_types", output_tensor_types_));
  input_tensor_requires_grads_ = info.GetAttrsOrDefault("input_tensor_requires_grads", std::vector<int64_t>());
  output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());
  SetPositions();
}

void PythonOpGradBase::SetPositions() {
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

void PythonOpGradBase::SetOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) const {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  auto outputs_count = static_cast<size_t>(ctx_internal->OutputCount());
  // It's possible that Pytorch returns None as gradient and ORT Python side may skip them.
  // In that case, returned_args may contain less arguments.
  outputs_count = outputs_count > returned_ortvalues.size() ? returned_ortvalues.size() : outputs_count;
  for (size_t i = 0; i < outputs_count; ++i) {
    ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(i, returned_ortvalues.at(i)));
  }
}

}  // namespace contrib
}  // namespace onnxruntime
