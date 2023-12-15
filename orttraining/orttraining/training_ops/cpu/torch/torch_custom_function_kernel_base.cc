// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ENABLE_TRAINING_TORCH_INTEROP

#include <iostream>
#include <chrono>
#include <thread>

#include "orttraining/core/framework/torch/python_common.h"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel_context_internal.h"
#endif
#include "orttraining/core/framework/torch/custom_function_register.h"
#include "orttraining/core/framework/torch/torch_proxy.h"
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"

using namespace onnxruntime::language_interop_ops::torch;

namespace onnxruntime {
namespace contrib {

namespace {

std::string GetInvokeIdString(const void* ptr) {
  const auto now = std::chrono::high_resolution_clock::now();
  std::ostringstream oss;
  oss << std::this_thread::get_id() << "-"
      << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()
      << "-" << std::rand() % 1000000 << "-" << reinterpret_cast<uint64_t>(ptr);
  return oss.str();
}

std::vector<std::optional<OrtValue>> CreateOrtValueArgs(OpKernelContext* context,
                                                        const int begin_index,
                                                        const int num_arg) {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  std::vector<std::optional<OrtValue>> args;
  for (int i = 0; i < num_arg; ++i) {
    int input_index = begin_index + i;
    if (context->Input<Tensor>(input_index)) {
      args.push_back(*ctx_internal->GetInputMLValue(input_index));
    } else {  // if the grad input is not provided.
      args.push_back(std::nullopt);
    }
  }
  return args;
}
}  // namespace

void PythonOpBase::Init(const OpKernelInfo& info) {
  ORT_THROW_IF_ERROR(info.GetAttr("func_name", &name_));

  is_training_mode_ = static_cast<bool>(info.GetAttrOrDefault("training_mode", static_cast<int64_t>(0)));

  safe_run_mode_enabled_ = static_cast<bool>(info.GetAttrOrDefault("safe_run_mode", static_cast<int64_t>(1)));

  ORT_THROW_IF_ERROR(info.GetAttr("input_convention", &input_convention_));

  input_requires_grads_ = info.GetAttrsOrDefault(
      "input_requires_grads", std::vector<int64_t>(input_convention_.size(), 0));

  // Input tensors.
  ORT_THROW_IF_ERROR(info.GetAttrs("input_tensor_types", input_tensor_types_));

  ORT_ENFORCE(input_tensor_types_.size() == info.node().InputDefs().size());

  // Input bool scalars.
  input_bool_scalars_ = info.GetAttrsOrDefault("input_bool_scalars", std::vector<int64_t>());
  input_bool_scalar_positions_ = info.GetAttrsOrDefault("input_bool_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_bool_scalars_.size() == input_bool_scalar_positions_.size());

  // Input int scalars.
  input_int_scalars_ = info.GetAttrsOrDefault("input_int_scalars", std::vector<int64_t>());
  input_int_scalar_positions_ = info.GetAttrsOrDefault("input_int_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_int_scalars_.size() == input_int_scalar_positions_.size());

  // Input float scalars.
  input_float_scalars_ = info.GetAttrsOrDefault("input_float_scalars", std::vector<float>());
  input_float_scalar_positions_ = info.GetAttrsOrDefault("input_float_scalar_positions", std::vector<int64_t>());

  ORT_ENFORCE(input_float_scalars_.size() == input_float_scalar_positions_.size());

  // Input bool tuples.
  input_bool_tuples_ = info.GetAttrsOrDefault("input_bool_tuples", std::vector<int64_t>());
  input_bool_tuple_positions_ = info.GetAttrsOrDefault("input_bool_tuple_positions", std::vector<int64_t>());
  input_bool_tuple_begins_ = info.GetAttrsOrDefault("input_bool_tuple_begins", std::vector<int64_t>());

  ORT_ENFORCE(input_bool_tuple_positions_.size() == input_bool_tuple_begins_.size());

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
  auto non_tensor_input_count = input_bool_scalars_.size() + input_int_scalars_.size() +
                                input_float_scalars_.size() +
                                input_bool_tuple_positions_.size() +
                                input_int_tuple_positions_.size() +
                                input_float_tuple_positions_.size() +
                                input_pointer_scalars_.size();
  ORT_ENFORCE(non_tensor_input_count + input_tensor_types_.size() == input_convention_.size(),
              "Total input (tensor + non-tensor) count did not match.");

  // Output tensors.
  ORT_THROW_IF_ERROR(info.GetAttrs("output_tensor_types", output_tensor_types_));

  all_output_to_tensor_input_reuse_map_ =
      info.GetAttrsOrDefault("tensor_reuse_map", std::vector<int64_t>((info.node().OutputDefs().size()), -1));

  CreateConstArgs();
  CreateArgPositions();

  kernel_invoke_id_ = GetInvokeIdString(this);
}

void PythonOpBase::Clear() {
  for (const auto& arg : const_arg_set_.GetArgs()) {
    // Only release owned PyObject.
    if (arg.is_owned) {
      auto obj = reinterpret_cast<PyObject*>(arg.data_ptr);
      Py_DECREF(obj);
    }
  }
}

void PythonOpBase::RunForward(OpKernelContext* context,
                              void** diff_ctx,
                              std::vector<OrtValue>& returned_ortvalues) const {
  // Create non-constant arguments for calling Python function.
  // Constant arguments are created in ctor.
  std::vector<std::optional<OrtValue>> args = CreateOrtValueArgs(context, 0, context->InputCount());
  // Invoke Python calls.
  TorchProxy::GetInstance().Forward(
      name_,
      safe_run_mode_enabled_ ? OrtTorchFunctionPool::GetInstance().GetForwardCore(name_)
                             : OrtTorchFunctionPool::GetInstance().GetUnsafeForwardCore(name_),
      input_requires_grads_,
      args,
      arg_positions_,
      const_arg_set_.GetDataPtrs(),
      const_arg_set_.GetPositions(),
      is_training_mode_,
      all_output_to_tensor_input_reuse_map_,
      kernel_invoke_id_,
      safe_run_mode_enabled_,
      diff_ctx,
      returned_ortvalues);

  const size_t returned_output_count = 1 + returned_ortvalues.size();
  const size_t kernel_output_count = static_cast<size_t>(context->OutputCount());
  ORT_ENFORCE(returned_output_count == kernel_output_count, "Output count mismatch for PythonOp run, ",
              "returned_output_count: ", returned_output_count, ", expected kernel_output_count: ",
              kernel_output_count);
}

void PythonOpBase::SetOutputs(OpKernelContext* context, void* diff_ctx, std::vector<OrtValue>& returned_args) const {
  SetContextOutput(context, diff_ctx);
  SetOtherOutputs(context, returned_args);
}

void PythonOpBase::AddPrimitiveTypeScalarArgs() {
  for (size_t i = 0; i < input_bool_scalars_.size(); ++i) {
    const_arg_set_.Add(input_bool_scalar_positions_.at(i), PyBool_FromLong(input_bool_scalars_.at(i)), true /*owned*/);
  }

  for (size_t i = 0; i < input_int_scalars_.size(); ++i) {
    const_arg_set_.Add(input_int_scalar_positions_.at(i),
                       Py_BuildValue("L", static_cast<long long>(input_int_scalars_.at(i))),
                       true /*owned*/);
  }

  for (size_t i = 0; i < input_float_scalars_.size(); ++i) {
    const_arg_set_.Add(input_float_scalar_positions_.at(i), Py_BuildValue("f", input_float_scalars_.at(i)),
                       true /*owned*/);
  }
}

void PythonOpBase::AddInputTupleArgs() {
  for (size_t i = 0; i < input_bool_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_bool_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end =
        (i + 1 == input_bool_tuple_begins_.size()) ? input_bool_tuples_.size() : input_bool_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = PyBool_FromLong(input_bool_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }

    const_arg_set_.Add(input_bool_tuple_positions_.at(i), tuple, true /*owned*/);
  }

  for (size_t i = 0; i < input_int_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_int_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end =
        (i + 1 == input_int_tuple_begins_.size()) ? input_int_tuples_.size() : input_int_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("L", input_int_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }

    const_arg_set_.Add(input_int_tuple_positions_.at(i), tuple, true /*owned*/);
  }
}

void PythonOpBase::AddFloatTupleArgs() {
  for (size_t i = 0; i < input_float_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_float_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end =
        (i + 1 == input_float_tuple_begins_.size()) ? input_float_tuples_.size() : input_float_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("f", input_float_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }

    const_arg_set_.Add(input_float_tuple_positions_.at(i), tuple, true /*owned*/);
  }
}

void PythonOpBase::AddPointerScalarArgs() {
  for (size_t i = 0; i < input_pointer_scalars_.size(); ++i) {
    PyObject* ptr = reinterpret_cast<PyObject*>(input_pointer_scalars_.at(i));
    // We don't want to own the Python object from C++ side because once C++ destructor called through pybind,
    // it may trigger python side object destroying, potentially requires GILs, resulting in a hang.
    // Instead, we have mechanism during exporting we increase the reference count already.
    const_arg_set_.Add(input_pointer_scalar_positions_.at(i), ptr, false /*owned*/);
  }
}

void PythonOpBase::CreateConstArgs() {
  ORT_ENFORCE(const_arg_set_.Size() == 0);
  AddPrimitiveTypeScalarArgs();
  AddInputTupleArgs();
  AddFloatTupleArgs();
  AddPointerScalarArgs();

  // Freeze the constant arg.
  const_arg_set_.Finalize();
}

void PythonOpBase::CreateArgPositions() {
  ORT_ENFORCE(arg_positions_.size() == 0);

  // occupied[i] being true means the i-th input argument
  // to Python function has been set.
  std::vector<bool> occupied(input_tensor_types_.size() + const_arg_set_.Size(), false);

  // We know all non-tensors were set above, so let's catch up.
  for (auto& arg : const_arg_set_.GetArgs()) {
    occupied.at(arg.position) = true;
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
  ORT_ENFORCE(returned_ortvalues.size() == all_output_to_tensor_input_reuse_map_.size() - 1,
              "PythonOp output count mismatch inplace map count.",
              returned_ortvalues.size(), " != ", all_output_to_tensor_input_reuse_map_.size() - 1);
  for (size_t i = 0; i < returned_ortvalues.size(); ++i) {
    size_t output_index = i + 1;
    if (all_output_to_tensor_input_reuse_map_[output_index] != -1) {
      const void* tensor_address = returned_ortvalues[i].Get<Tensor>().DataRaw();
      const void* input_tensor_address =
          context->Input<Tensor>(all_output_to_tensor_input_reuse_map_[output_index])->DataRaw();
      ORT_ENFORCE(tensor_address == input_tensor_address,
                  "PythonOp inplace tensor address mismatch, output index: ", output_index, ", input index: ",
                  all_output_to_tensor_input_reuse_map_[output_index]);
    }

    // Notes: if the buffer is created, managed by PyTorch, converted to OrtValue through dlpack here,
    // but also be used outside ORT later, we don't need to be concerned about
    // "when the buffer of returned_ortvalues[i] is erased by ORT during releasing that OrtValue causing
    //  the PyTorch code still using that buffer will be failed".
    // In this case, the created OrtValue's destructor will not release the buffer,
    // instead it will release a tensor pointing to that buffer, where PyTorch will decide whether to release
    // the buffer or not, if the tensor storage is not used by any other tensors
    // (https://github.com/PyTorch/PyTorch/blob/ac603bc2f8ffac8fc061cfb99e77537464da4b18/aten/src/ATen/DLConvertor.cpp#L257C25-L257C29).
    ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(static_cast<int>(i + 1), returned_ortvalues[i]));
  }
}

void PythonOpGradBase::Init(const OpKernelInfo& info) {
  ORT_THROW_IF_ERROR(info.GetAttr("func_name", &name_));
  ORT_THROW_IF_ERROR(info.GetAttrs("input_tensor_types", input_tensor_types_));
  ORT_THROW_IF_ERROR(info.GetAttr("output_convention", &output_convention_));
  ORT_THROW_IF_ERROR(info.GetAttrs("output_tensor_types", output_tensor_types_));
  output_tensor_requires_grads_ = info.GetAttrsOrDefault("output_tensor_requires_grads", std::vector<int64_t>());
  ORT_ENFORCE(output_tensor_types_.size() == output_tensor_requires_grads_.size(),
              "backward tensor output count mismatch");
  safe_run_mode_enabled_ = static_cast<bool>(info.GetAttrOrDefault("safe_run_mode", static_cast<int64_t>(1)));
  std::vector<int64_t> tensor_output_to_tensor_input_alias_map =
      info.GetAttrsOrDefault("tensor_reuse_map",
                             std::vector<int64_t>((info.node().OutputDefs().size()), -1));
  all_output_to_tensor_input_reuse_map_.clear();
  all_output_to_tensor_input_reuse_map_.reserve(output_convention_.size());
  size_t tensor_output_index = 0;
  for (size_t i = 0; i < output_convention_.size(); ++i) {
    if (output_convention_[i] == 'd') {
      all_output_to_tensor_input_reuse_map_.push_back(
          tensor_output_to_tensor_input_alias_map[tensor_output_index] == -1
              ? -1
              : tensor_output_to_tensor_input_alias_map[tensor_output_index]);
      ++tensor_output_index;
    } else {
      all_output_to_tensor_input_reuse_map_.push_back(-1);
    }
  }

  SetPositions();

  kernel_invoke_id_ = GetInvokeIdString(this);
}

void PythonOpGradBase::RunBackward(OpKernelContext* context,
                                   std::vector<OrtValue>& returned_ortvalues) const {
  std::vector<std::optional<OrtValue>> args = CreateOrtValueArgs(context, 1, context->InputCount() - 1);
  // This is called "const" because that's how PyTorch calls all non-tensor inputs.
  const Tensor* context_id_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(context_id_tensor, "Context ID (first input) should not be null.");
  const int64_t* context_index_ptr = context_id_tensor->template Data<int64_t>();
  void* ctx_ptr = OrtTorchFunctionPool::GetInstance().GetContext(*context_index_ptr);
  auto const_args = {ctx_ptr};

  std::string err;
  TorchProxy::GetInstance().Backward(
      name_,
      OrtTorchFunctionPool::GetInstance().GetBackwardCore(name_),
      args,
      arg_positions_,
      const_args,
      const_arg_positions_,
      all_output_to_tensor_input_reuse_map_,
      kernel_invoke_id_,
      safe_run_mode_enabled_,
      returned_ortvalues);

  OrtTorchFunctionPool::GetInstance().UnregisterContext(*context_index_ptr);
}

void PythonOpGradBase::SetOutputs(OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) const {
  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  ORT_ENFORCE(output_convention_.size() == returned_ortvalues.size(), "backward output count mismatch.");
  int tensor_output_index = 0;
  for (size_t i = 0; i < returned_ortvalues.size(); ++i) {
    if (output_convention_[i] == 'd') {
      if (output_tensor_requires_grads_[tensor_output_index]) {
        if (all_output_to_tensor_input_reuse_map_[i] != -1) {
          const Tensor* input_tensor = context->Input<Tensor>(all_output_to_tensor_input_reuse_map_[i]);
          if (input_tensor) {
            ORT_ENFORCE(input_tensor, "PythonOpGrad input tensor should not be null. input index: ", all_output_to_tensor_input_reuse_map_[i]);

            // Be noted: PythonOpGrad's input won't be non-tensor.
            ORT_ENFORCE(all_output_to_tensor_input_reuse_map_[i] < context->InputCount(), "PythonOpGrad inplace tensor index out of bound.");
            const void* tensor_address = returned_ortvalues[i].Get<Tensor>().DataRaw();

            const void* input_tensor_address = input_tensor->DataRaw();
            ORT_ENFORCE(tensor_address == input_tensor_address,
                        "PythonOpGrad inplace tensor address mismatch, output index: ", i, ", input index: ", all_output_to_tensor_input_reuse_map_[i]);
          }
        }

        // Notes: if the buffer is created, managed by PyTorch, converted to OrtValue through dlpack here,
        // but also be used outside ORT later, we don't need to be concerned about
        // "when the buffer of returned_ortvalues[i] is erased by ORT during releasing that OrtValue causing
        //  the PyTorch code still using that buffer will be failed".
        // In this case, the created OrtValue's destructor will not release the buffer,
        // instead it will release a tensor pointing to that buffer, where PyTorch will decide whether to release
        // the buffer or not, if the tensor storage is not used by any other tensors
        // (https://github.com/PyTorch/PyTorch/blob/ac603bc2f8ffac8fc061cfb99e77537464da4b18/aten/src/ATen/DLConvertor.cpp#L257C25-L257C29).
        ORT_THROW_IF_ERROR(ctx_internal->SetOutputMLValue(tensor_output_index, returned_ortvalues.at(i)));
      }
      ++tensor_output_index;
    }
  }

  ORT_ENFORCE(tensor_output_index == ctx_internal->OutputCount(), "backward tensor output count mismatch.");
}

void PythonOpGradBase::SetPositions() {
  ORT_ENFORCE(const_arg_positions_.size() == 0);
  ORT_ENFORCE(arg_positions_.size() == 0);

  // PyTorch's autograd context is the first (indexed by 0) input of the called Python function.
  // Note that here we will call autograd.Function.backward(ctx, tensor0, tensor1, ...).
  const_arg_positions_ = {0};

  // The rest inputs are just PyTorch tensors.
  arg_positions_.resize(input_tensor_types_.size());
  for (size_t i = 0; i < arg_positions_.size(); ++i) {
    // i-th tensor is the (i+1)-th input of autograd.Function.backward.
    arg_positions_.at(i) = static_cast<int64_t>(i) + 1;
  }
}

}  // namespace contrib
}  // namespace onnxruntime

#endif
