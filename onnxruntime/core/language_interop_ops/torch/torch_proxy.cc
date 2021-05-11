// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#include "core/framework/tensorprotoutils.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/object_pointer.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/platform/env.h"
#include "core/util/dlpack_convertor.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

template <>
void ObjectPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  if (dlmanged_tensor) {
    // the dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // the dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

bool ExtractPointerOutput(PyObject* pyObj, std::vector<void*>& outputs) {
  void* prt = PyLong_AsVoidPtr(pyObj);
  // std::cout << "ExtractPointerOutput:" << prt << std::endl;
  outputs.push_back(prt);
  return true;
}

TorchProxy& TorchProxy::GetInstance() {
  static TorchProxy proxy;
  return proxy;
}

int32_t TorchProxy::GetGil() const {
  return PyGILState_Ensure();
}

void TorchProxy::PutGil(int32_t state) const {
  PyGILState_Release((PyGILState_STATE)state);
}

void CheckArguments(
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*> obj_args,
    const std::vector<int64_t>& obj_indices) {
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
}

// len: the number of input arguments.
// tensor_indices: if tensor_indices[i] is j,
//                 then the j-th input argument should be a tensor.
PyObject* CreateTensorFlags(
    const size_t len,
    const std::vector<int64_t>& tensor_indices) {
  PyObject* flags = PyList_New(len);

  // First we fill the list with 0. Later we will
  // assign 1's to tensors' corresponding positions.
  for (size_t i = 0; i < len; ++i) {
    PyObject* zero = PyLong_FromLong(0);
    PyList_SetItem(flags, i, zero);
  }

  for (const auto i : tensor_indices) {
    PyObject* one = PyLong_FromLong(1);
    PyList_SetItem(flags, i, one);
  }

  return flags;
}

PyObject* CreateRequiresGradFlags(
    const std::vector<int64_t>& requires_grads) {
  PyObject* flags = PyList_New(requires_grads.size());
  for (size_t i = 0; i < requires_grads.size(); ++i) {
    PyObject* value;
    if (requires_grads.at(i) != 0) {
      value = PyLong_FromLong(1);
    } else {
      value = PyLong_FromLong(0);
    }
    PyList_SetItem(flags, i, value);
  }
  return flags;
}

void InvokeRunner(
    PyObject* callback_runner,
    PyObject* args,
    std::vector<void*>& returned_args) {
  PyObject* result = PyObject_CallObject(reinterpret_cast<PyObject*>(callback_runner), args);
  if (PyErr_Occurred()) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }

  ORT_ENFORCE(PyTuple_Check(result), "Python function must return a tuple.");
  for (int i = 0; i < PyTuple_Size(result); ++i) {
    ORT_ENFORCE(ExtractPointerOutput(PyTuple_GetItem(result, i), returned_args));
  }
}

PyObject* CreateForwardArguments(
    PyObject* callback,
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    bool is_training_mode) {
  ORT_ENFORCE(PyCallable_Check(callback), "Forward callback is not callable.");
  int64_t num_args_without_inputs = 4;
  PyObject* args = PyTuple_New(num_args_without_inputs + len);
  PyObject* tensor_flags = CreateTensorFlags(len, tensor_indices);
  PyObject* requires_grad_flags = CreateRequiresGradFlags(requires_grads);
  Py_INCREF(callback);
  PyTuple_SetItem(args, 0, callback);
  Py_INCREF(requires_grad_flags);
  PyTuple_SetItem(args, 1, requires_grad_flags);
  Py_INCREF(tensor_flags);
  PyTuple_SetItem(args, 2, tensor_flags);

  if (is_training_mode) {
    Py_INCREF(Py_True);
  } else {
    Py_INCREF(Py_False);
  }
  PyTuple_SetItem(args, 3, is_training_mode ? Py_True : Py_False);

  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*tensor_args[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    Py_INCREF(dltensor);
    PyTuple_SetItem(args, num_args_without_inputs + tensor_indices[i], dltensor);
  }

  for (size_t i = 0; i < obj_args.size(); ++i) {
    PyObject* pyobj = reinterpret_cast<PyObject*>(obj_args[i]);
    Py_INCREF(pyobj);
    PyTuple_SetItem(args, num_args_without_inputs + obj_indices[i], pyobj);
  }

  return args;
}

void Invoke(
    PyObject* runner,
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<void*>& returned_args,
    bool is_training_mode) {
  const auto len = tensor_args.size() + obj_args.size();
  CheckArguments(len, requires_grads, tensor_args, tensor_indices, obj_args, obj_indices);
  PythonObjectPtr args_ptr = PythonObjectPtr(CreateForwardArguments(
      callback,
      len,
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      is_training_mode));
  InvokeRunner(runner, args_ptr.get(), returned_args);
}

void TorchProxy::Forward(
    void* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<void*>& returned_args,
    bool is_training_mode) {
  auto runner = OrtTorchFunctionPool::GetInstance().GetForwardRunner();
  Invoke(
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      returned_args,
      is_training_mode);
}

void TorchProxy::Backward(
    void* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<void*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<void*>& returned_args) {
  auto runner = OrtTorchFunctionPool::GetInstance().GetBackwardRunner();
  Invoke(
      runner,
      reinterpret_cast<PyObject*>(callback),
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      returned_args,
      true /*is_training_mode*/);
}
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime