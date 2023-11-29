// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/torch/torch_proxy.h"
#include "orttraining/core/framework/torch/python_common.h"
#include "orttraining/core/framework/torch/dlpack_python.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/core/framework/torch/custom_function_register.h"
#include "orttraining/core/framework/torch/refcount_tracker.h"
#include "orttraining/core/framework/torch/gil.h"
#include "core/platform/env.h"

namespace onnxruntime::language_interop_ops::torch {

void PythonObjectDeleter(PyObject* ptr) { Py_XDECREF(ptr); };

PyObject* Ort_PyTuple_New(const size_t len, const std::string& log_tag) {
  PyObject* item = PyTuple_New(len);
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  return item;
}

void Ort_PyTuple_SetItem_Incref(PyObject* py_tuple, size_t index, PyObject* item, const std::string& log_tag) {
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  Py_INCREF(item);
  PyTuple_SetItem(py_tuple, index, item);
}

void Ort_PyTuple_SetItem_NoIncref(PyObject* py_tuple, size_t index, PyObject* item, const std::string& log_tag) {
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  PyTuple_SetItem(py_tuple, index, item);
}

PyObject* Ort_PyList_New(const size_t len, const std::string& log_tag) {
  PyObject* item = PyList_New(len);
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  return item;
}

void Ort_PyList_SetItem_Incref(PyObject* py_list, size_t index, PyObject* item, const std::string& log_tag) {
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  Py_INCREF(item);
  PyList_SetItem(py_list, index, item);
}

void Ort_PyList_SetItem_NoIncref(PyObject* py_list, size_t index, PyObject* item, const std::string& log_tag) {
  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::PythonCallArgs, item, log_tag);
  PyList_SetItem(py_list, index, item);
}

void CheckArguments(
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<std::optional<OrtValue>>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*> obj_args,
    const std::vector<int64_t>& obj_indices) {
#ifndef NDEBUG
  ORT_UNUSED_PARAMETER(len);
  ORT_UNUSED_PARAMETER(requires_grads);
  ORT_UNUSED_PARAMETER(tensor_args);
  ORT_UNUSED_PARAMETER(tensor_indices);
  ORT_UNUSED_PARAMETER(obj_args);
  ORT_UNUSED_PARAMETER(obj_indices);
#else
  ORT_ENFORCE(requires_grads.size() == len);
  ORT_ENFORCE(tensor_args.size() + obj_args.size() == len);
  ORT_ENFORCE(tensor_args.size() == tensor_indices.size());
  ORT_ENFORCE(obj_args.size() == obj_indices.size());

  for (const auto i : requires_grads) {
    ORT_ENFORCE(i == 0 || i == 1,
                "Flag of requiring gradient must be either 0 (not required) or 1 (required) but got ", i);
  }

  std::vector<int64_t> counts(len, 0);

  for (const auto i : tensor_indices) {
    ORT_ENFORCE(i >= 0 && static_cast<size_t>(i) < len, "Index range is from 0 to ", len - 1, ", but found ", i);
    counts.at(i) += 1;
  }
  for (const auto i : obj_indices) {
    ORT_ENFORCE(i >= 0 && static_cast<size_t>(i) < len, "Index range is from 0 to ", len - 1, ", but found ", i);
    counts.at(i) += 1;
  }
  for (size_t i = 0; i < len; ++i) {
    ORT_ENFORCE(counts.at(i) == 1, "Duplicated or unused argument index detected: ", i);
  }
#endif
}

// len: the number of input arguments.
// tensor_indices: if tensor_indices[i] is j,
//                 then the j-th input argument should be a tensor.
PyObject* CreateTensorFlags(
    const size_t len,
    const std::vector<int64_t>& tensor_indices) {
  PyObject* flags = Ort_PyList_New(len, "tensor_flags_list");

  // First we fill the list with 0. Later we will
  // assign 1's to tensors' corresponding positions.
  for (size_t i = 0; i < len; ++i) {
    PyObject* zero = PyLong_FromLong(0);
    Ort_PyList_SetItem_NoIncref(flags, i, zero, std::to_string(__LINE__));
  }

  for (const auto i : tensor_indices) {
    PyObject* one = PyLong_FromLong(1);
    Ort_PyList_SetItem_NoIncref(flags, i, one, std::to_string(__LINE__));
  }

  return flags;
}

// flags[i] corresponds to the i-th input of apply/backward.
PyObject* CreateRequiresGradFlags(
    const std::vector<int64_t>& requires_grads) {
  PyObject* flags = Ort_PyList_New(requires_grads.size(), "require_grads_list");
  for (size_t i = 0; i < requires_grads.size(); ++i) {
    PyObject* value;
    if (requires_grads.at(i) != 0) {
      value = Py_True;
    } else {
      value = Py_False;
    }
    Ort_PyList_SetItem_Incref(flags, i, value, std::to_string(__LINE__));
  }
  return flags;
}

PyObject* CreateInplaceMap(
    const std::vector<int64_t>& inplace_map) {
  PyObject* inplace_map_obj = Ort_PyList_New(inplace_map.size(), "inplace_map");

  for (size_t output_index = 0; output_index < inplace_map.size(); ++output_index) {
    PyObject* input_index = PyLong_FromLong(inplace_map[output_index]);
    Ort_PyList_SetItem_NoIncref(inplace_map_obj, output_index, input_index, std::to_string(__LINE__));
  }

  return inplace_map_obj;
}

void InvokeRunner(
    PyObject* callback_runner,
    PyObject* args,
    bool is_training_mode,
    void** diff_ctx,
    std::vector<OrtValue>& returned_ortvalues) {
  PythonObjectPtr result_ptr(PyObject_CallObject(callback_runner, args), PythonObjectDeleter);

  if (PyErr_Occurred()) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }

  ORT_ENFORCE(PyTuple_Check(result_ptr.get()), "Python function must return a tuple.");

  size_t i = 0;
  if (diff_ctx) {
    // Assume that the first input element in the returned tuple is autograd context
    // from Pytorch.
    PyObject* py_obj = PyTuple_GetItem(result_ptr.get(), 0);
    if (is_training_mode) {
      if (py_obj == Py_None) {
        LOGS_DEFAULT(VERBOSE) << "Under training mode, autograd context found to be Py_None.";
      } else {
        const auto refcnt = Py_REFCNT(py_obj);
        // We don't need do ref increase here because, python returns tensor.grad_fn as part of
        // tuple, who increased the refcnt already (and tensor persist until the backward kernels completed).
        // Pytorch also increases refcnt before apply() return, so we should expect refcount >= 2.
        // We say "at least" 2 because user could increase the context refcnt as well in their autograd forward()
        // and backward() functions.
        ORT_ENFORCE(refcnt >= 2, "Ref count of context should be 2, but actually it's ", refcnt, ".");
        if (refcnt > 2) {
          LOGS_DEFAULT(VERBOSE) << "Autograd context refcnt > 2, refcnt: " << refcnt;
        }
      }
    } else {
      ORT_ENFORCE(py_obj == Py_None, "Under inference mode, autograd context should be Py_None.");
    }
    *diff_ctx = py_obj;
    ++i;
  }

  // i is 1 if the first element is autograd context. Otherwise, i is 0, so we read from the
  // first element.
  for (; i < static_cast<size_t>(PyTuple_Size(result_ptr.get())); ++i) {
    PyObject* dl_tensor_pointer = PyTuple_GetItem(result_ptr.get(), i);
    if (dl_tensor_pointer == Py_None) {
      OrtValue empty_ort_value;
      returned_ortvalues.push_back(empty_ort_value);
    } else {
      ORT_ENFORCE(Py_REFCNT(dl_tensor_pointer) == 1, "Ref count of dl_tensor_pointer should be 1.");
      // Todo (pengwa): be noted we did not pass whether tensor is bool or not.
      // Currently we assume we don't pass boolean data.
      returned_ortvalues.push_back(training::framework::torch::FromDlpack(dl_tensor_pointer, false));
    }
  }
}

PythonObjectPtr CreatePythonCallArguments(
    PyObject* callback,
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<std::optional<OrtValue>>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    const bool is_training_mode,
    const std::vector<int64_t>& inplace_map,
    const std::string& invoke_id,
    const std::string& func_name) {
  ORT_ENFORCE(PyCallable_Check(callback), "Forward callback is not callable.");
  // The number of variables before those of
  // autograd.Function.apply and autograd.Function.backward.
  // The extra variables are used to configure the launch
  // forward and backward runners.
  constexpr int64_t num_control_args = 7;

  // All arguments created for Python call will be destroyed along with PythonObjectPtr.
  PythonObjectPtr args(Ort_PyTuple_New(num_control_args + len, "forward_arguments_tuple"), PythonObjectDeleter);
  PyObject* tensor_flags = CreateTensorFlags(len, tensor_indices);
  PyObject* requires_grad_flags = CreateRequiresGradFlags(requires_grads);

  Ort_PyTuple_SetItem_Incref(args.get(), 0, callback, "callback_function");
  Ort_PyTuple_SetItem_NoIncref(args.get(), 1, requires_grad_flags, "requires_grad_flags");
  Ort_PyTuple_SetItem_NoIncref(args.get(), 2, tensor_flags, "tensor_flags");
  PyObject* is_training_mode_arg = is_training_mode ? Py_True : Py_False;
  Ort_PyTuple_SetItem_Incref(args.get(), 3, is_training_mode_arg, "is_training_mode");

  PyObject* inplace_map_arg = CreateInplaceMap(inplace_map);
  Ort_PyTuple_SetItem_NoIncref(args.get(), 4, inplace_map_arg, "inplace_map");

  PyObject* kernel_invoke_id_arg = PyBytes_FromStringAndSize(invoke_id.c_str(), invoke_id.size());
  Ort_PyTuple_SetItem_NoIncref(args.get(), 5, kernel_invoke_id_arg, "kernel_invoke_id_arg");

  PyObject* func_name_arg = PyBytes_FromStringAndSize(func_name.c_str(), func_name.size());
  Ort_PyTuple_SetItem_NoIncref(args.get(), 6, func_name_arg, "func_name_arg");

  // Tensor inputs to call autograd.Function.apply or autograd.Function.backward.
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    if (!tensor_args[i].has_value()) {
      Ort_PyTuple_SetItem_Incref(args.get(), num_control_args + tensor_indices[i], Py_None,
                                 "non_tensor_args");
      continue;
    }

    // Wrap with DLPack, then transfer to Python for its release.
    PyObject* dl_tensor = training::framework::torch::ToDlpack(tensor_args[i].value());
    Ort_PyTuple_SetItem_NoIncref(args.get(), num_control_args + tensor_indices[i], dl_tensor,
                                 "dltensor");
  }

  // Non-tensor inputs to call autograd.Function.apply or autograd.Function.backward.
  for (size_t i = 0; i < obj_args.size(); ++i) {
    PyObject* pyobj = reinterpret_cast<PyObject*>(obj_args[i]);
    Ort_PyTuple_SetItem_Incref(args.get(), num_control_args + obj_indices[i], pyobj,
                               "const_args");
  }

  return args;
}

void Invoke(
    const std::string& func_name,
    PyObject* runner,
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<std::optional<OrtValue>>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    const bool is_training_mode,
    const std::vector<int64_t>& inplace_map,
    const std::string& invoke_id,
    void** diff_ctx,
    std::vector<OrtValue>& returned_ortvalues) {
  const auto len = tensor_args.size() + obj_args.size();
  CheckArguments(len, requires_grads, tensor_args, tensor_indices, obj_args, obj_indices);
  RefCountTracker::GetInstance().Reset();
  {
    PythonObjectPtr args = CreatePythonCallArguments(
        callback,
        len,
        requires_grads,
        tensor_args,
        tensor_indices,
        obj_args,
        obj_indices,
        is_training_mode,
        inplace_map,
        invoke_id,
        func_name);

    RefCountTracker::GetInstance().DumpDetails("Before Invoke Python Call");
    InvokeRunner(runner, args.get(), is_training_mode, diff_ctx, returned_ortvalues);
  }

  RefCountTracker::GetInstance().DumpDetails("After Python Call Completed");
}

void TorchProxy::Forward(
    const std::string& func_name,
    void* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<std::optional<OrtValue>>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    const bool is_training_mode,
    const std::vector<int64_t>& inplace_map,
    const std::string& invoke_id,
    void** diff_ctx,
    std::vector<OrtValue>& returned_ortvalues) {
  // Semantically, this lock uniquely takes the ownership of TorchProxy
  // so that there will be only one of TorchProxy::Forward TorchProxy::Backward
  // can be run at one time.
  std::lock_guard<std::mutex> lock(mutex_);
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  auto runner = OrtTorchFunctionPool::GetInstance().GetForwardRunner();
  Invoke(
      func_name,
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      is_training_mode,
      inplace_map,
      invoke_id,
      diff_ctx,
      returned_ortvalues);
}

void TorchProxy::Backward(
    const std::string& func_name,
    void* callback,
    const std::vector<std::optional<OrtValue>>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    const std::vector<int64_t>& inplace_map,
    const std::string& invoke_id,
    std::vector<OrtValue>& returned_ortvalues) {
  // Semantically, this lock uniquely takes the ownership of TorchProxy
  // so that there will be only one of TorchProxy::Forward TorchProxy::Backward
  // can be run at one time.
  std::lock_guard<std::mutex> lock(mutex_);
  // Python-related calls should happen only if guard is alive.
  GilGuard guard;
  auto runner = OrtTorchFunctionPool::GetInstance().GetBackwardRunner();

  // Pass all zero since backward inputs don't require gradients.
  const auto all_input_count = tensor_args.size() + obj_args.size();
  const std::vector<int64_t> requires_grads(all_input_count, 0);
  Invoke(
      func_name,
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      true /* is_training_mode */,
      inplace_map,
      invoke_id,
      nullptr /* context to store */,
      returned_ortvalues);
}

void TorchProxy::RunInputAliasFunction(
    void* input_alias_function,
    const std::string& node_proto_str,
    std::vector<int64_t>& fw_output_to_input_alias_map,
    std::vector<int64_t>& bw_output_to_input_alias_map) {
  PyObject* input_alias_func = reinterpret_cast<PyObject*>(input_alias_function);
  ORT_ENFORCE(PyCallable_Check(input_alias_func), "input_alias_func is not callable.");

  // All arguments created for Python call will be destroyed along with PythonObjectPtr.
  PythonObjectPtr args(Ort_PyTuple_New(1, "input_alias_func_arguments_tuple"), PythonObjectDeleter);
  PyObject* node_proto_ptr_arg = PyBytes_FromStringAndSize(node_proto_str.c_str(), node_proto_str.size());
  Ort_PyTuple_SetItem_NoIncref(args.get(), 0, node_proto_ptr_arg, "node_proto_ptr_arg");

  PythonObjectPtr result_ptr(PyObject_CallObject(input_alias_func, args.get()), PythonObjectDeleter);
  if (PyErr_Occurred()) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }

  bool is_tuple = PyTuple_Check(result_ptr.get());
  bool is_list = PyList_Check(result_ptr.get());
  ORT_ENFORCE(is_tuple || is_list, "Python function must return a tuple or a list. is_tuple: ",
              is_tuple, ", is_list: ", is_list);
  Py_ssize_t ret_tuple_size =
      is_tuple ? PyTuple_Size(result_ptr.get()) : PyList_Size(result_ptr.get());
  ORT_ENFORCE(ret_tuple_size == 2, "Input alias function must return a tuple/list of size 2.");

  for (Py_ssize_t tuple_index = 0; tuple_index < ret_tuple_size; ++tuple_index) {
    PyObject* alias_map = is_tuple ? PyTuple_GetItem(result_ptr.get(), tuple_index)
                                   : PyList_GetItem(result_ptr.get(), tuple_index);

    std::vector<int64_t>& output_to_input_alias_map =
        tuple_index == 0 ? fw_output_to_input_alias_map : bw_output_to_input_alias_map;

    bool is_elem_tuple = PyTuple_Check(alias_map);
    bool is_elem_list = PyList_Check(alias_map);

    ORT_ENFORCE(is_elem_tuple || is_elem_list, "Input alias map must be a tuple or a list. is_elem_list: ",
                is_elem_list, ", is_elem_tuple: ", is_elem_tuple);
    Py_ssize_t output_count = is_elem_tuple ? PyTuple_Size(alias_map) : PyList_Size(alias_map);
    for (Py_ssize_t output_index = 0; output_index < output_count; ++output_index) {
      PyObject* input_index =
          is_elem_tuple ? PyTuple_GetItem(alias_map, output_index) : PyList_GetItem(alias_map, output_index);
      ORT_ENFORCE(PyLong_Check(input_index), "Alias input index must be an integer.");
      int64_t alias_index_int = PyLong_AsLongLong(input_index);
      output_to_input_alias_map.push_back(alias_index_int);
    }
  }
}

}  // namespace onnxruntime::language_interop_ops::torch
