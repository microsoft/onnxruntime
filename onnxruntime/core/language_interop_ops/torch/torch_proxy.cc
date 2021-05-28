// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/language_interop_ops/torch/torch_proxy.h"

#include <Python.h>
#include "core/dlpack/dlpack_converter.h"
#include "core/framework/tensorprotoutils.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"
#include "core/language_interop_ops/torch/gil.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// For handling temporary PyObject pointer newly created with Py_XXX APIs, here is our practice:
// Convention:
//     Wrap those PyObject* in format of "PythonObjectPtr(Py_XXX(), PythonObjectDeleter)".
// Explaination:
//     That means, for the PyObject* created by Py_XXX(), its refcnt will be decreased by one
//     in the PythonObjectDeleter which is triggered once lifetime of PythonObjectPtr instance
//     ends.
void PythonObjectDeleter(PyObject* ptr) { Py_XDECREF(ptr); }
using PythonObjectPtr = std::unique_ptr<PyObject, std::function<void(PyObject*)>>;

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
    const std::vector<OrtValue>& tensor_args,
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
    ORT_ENFORCE(i == 0 || i == 1, "Flag of requiring gradient must be either 0 (not required) or 1 (required) but got ", i);
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
      value = PyLong_FromLong(1);
    } else {
      value = PyLong_FromLong(0);
    }
    Ort_PyList_SetItem_NoIncref(flags, i, value, std::to_string(__LINE__));
  }
  return flags;
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
      const auto& refcnt = Py_REFCNT(py_obj);
      // we don't need do ref increase here because, python returns tensor.grad_fn as part of
      // tuple, who increased the refcnt already (and tensor persist until the backward kernels completed).
      // Pytorch also increases refcnt before apply() return, so we should expect refcount >= 2.
      // We say "at least" 2 because user could increase the context refcnt as well in their autograd forward()
      // and backward() functions.
      ORT_ENFORCE(refcnt >= 2, "Ref count of context should be 2, but actually it's ", refcnt, ".");
      if (refcnt > 2) {
        LOGS_DEFAULT(WARNING) << "Autograd context refcnt > 2";
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
    ORT_ENFORCE(Py_REFCNT(dl_tensor_pointer) == 1, "Ref count of dl_tensor_pointer should be 1.");
    // Todo: be noted we did not pass whether tensor is bool or not.
    // Currently we assume we don't pass boolean data.
    returned_ortvalues.push_back(dlpack::FromDlpack(dl_tensor_pointer, false));
  }
}

PythonObjectPtr CreatePythonCallArguments(
    PyObject* callback,
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    const bool is_training_mode,
    const bool is_inplace) {
  ORT_ENFORCE(PyCallable_Check(callback), "Forward callback is not callable.");
  // The number of variables before those of
  // autograd.Function.apply and autograd.Function.backward.
  // The extra variables are used to configure the launch
  // forward and backward runners.
  constexpr int64_t num_control_args = 5;

  // All arguments created for Python call will be destroyed along with PythonObjectPtr.
  PythonObjectPtr args(Ort_PyTuple_New(num_control_args + len, "forward_arguments_tuple"), PythonObjectDeleter);
  PyObject* tensor_flags = CreateTensorFlags(len, tensor_indices);
  PyObject* requires_grad_flags = CreateRequiresGradFlags(requires_grads);

  Ort_PyTuple_SetItem_Incref(args.get(), 0, callback, "callback_function");
  Ort_PyTuple_SetItem_NoIncref(args.get(), 1, requires_grad_flags, "requires_grad_flags");
  Ort_PyTuple_SetItem_NoIncref(args.get(), 2, tensor_flags, "tensor_flags");
  PyObject* is_training_mode_arg = is_training_mode ? Py_True : Py_False;
  Ort_PyTuple_SetItem_Incref(args.get(), 3, is_training_mode_arg, "is_training_mode");
  PyObject* is_inplace_arg = is_inplace ? Py_True : Py_False;
  Ort_PyTuple_SetItem_Incref(args.get(), 4, is_inplace_arg, "is_inplace_mode");

  // Tensor inputs to call autograd.Function.apply or autograd.Function.backward.
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.
    PyObject* dl_tensor = dlpack::ToDlpack(tensor_args[i]);
    Ort_PyTuple_SetItem_NoIncref(args.get(), num_control_args + tensor_indices[i], dl_tensor,
                                 "dltensor");
  }

  // Non-tensor inputs to call autograd.Function.apply or autograd.Function.backward.
  for (size_t i = 0; i < obj_args.size(); ++i) {
    PyObject* pyobj = reinterpret_cast<PyObject*>(obj_args[i]);
    Ort_PyTuple_SetItem_Incref(args.get(), num_control_args + obj_indices[i], pyobj,
                               "const_args");
  }

  return std::move(args);
}

void Invoke(
    PyObject* runner,
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    void** diff_ctx,
    std::vector<OrtValue>& returned_ortvalues,
    const bool is_training_mode,
    const bool is_inplace) {
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
        is_inplace);

    RefCountTracker::GetInstance().DumpDetails("Before Invoke Python Call");
    InvokeRunner(runner, args.get(), is_training_mode, diff_ctx, returned_ortvalues);
  }

  RefCountTracker::GetInstance().DumpDetails("After Python Call Completed");
}

void TorchProxy::Forward(
    void* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    void** diff_ctx,
    std::vector<OrtValue>& returned_ortvalues,
    const bool is_training_mode,
    const bool is_inplace) {
  // Python-related calls should happen only if guard is alive.
  std::lock_guard<std::mutex> lock(mutex_);
  GilGuard guard;
  auto runner = OrtTorchFunctionPool::GetInstance().GetForwardRunner();
  Invoke(
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      diff_ctx,
      returned_ortvalues,
      is_training_mode,
      is_inplace);
}

void TorchProxy::Backward(
    void* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<OrtValue>& returned_ortvalues,
    const bool is_inplace) {
  // Python-related calls should happen only if guard is alive.
  std::lock_guard<std::mutex> lock(mutex_);
  GilGuard guard;
  auto runner = OrtTorchFunctionPool::GetInstance().GetBackwardRunner();
  Invoke(
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      nullptr /* context to store */,
      returned_ortvalues,
      true /* is_training_mode */,
      is_inplace);
}
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
