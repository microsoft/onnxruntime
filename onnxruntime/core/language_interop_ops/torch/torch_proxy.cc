// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <Python.h>
#include "core/framework/tensorprotoutils.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "core/platform/env.h"
#include "core/util/dlpack_convertor.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {
class Scope {
 public:
  Scope(const std::vector<PyObject*>& objs = {}) : objs_(objs) {
    mtx_.lock();
  }
  ~Scope() {
    for (auto obj : objs_) {
      Py_XDECREF(obj);
    }
    mtx_.unlock();
  }
  void Add(PyObject* obj) {
    objs_.push_back(obj);
  }

 private:
  static std::mutex mtx_;
  std::vector<PyObject*> objs_;
};

std::mutex Scope::mtx_;

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

TorchProxy& TorchProxy::GetInstance() {
  static TorchProxy proxy;
  return proxy;
}

TorchProxy::TorchProxy() {
  Scope scope;
}

TorchProxy::~TorchProxy() {
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
    const std::vector<PyObject*> obj_args,
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

// Generate a list of flags, returning an owning reference to it.
//
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

// Generate a list of flags, returning an owning reference to it.  

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

// Invoke the underlying python method, returning a tuple of results.
// We borrow references to callback_runner and args.  We are own the
// result tuple, and release our ref count on it prior to returning.

void InvokeRunner(PyObject* callback_runner, 
		  PyObject* args,
		  std::vector<OrtValue>& returned_args) {
  PyObject* result = PyObject_CallObject(callback_runner, args);
  if (PyErr_Occurred()) {
    PyErr_Print();
    ORT_THROW("Python function execution fails with the above information.");
  }

  ORT_ENFORCE(PyTuple_Check(result), "Python function must return a tuple.");
  for (int i = 0; i < PyTuple_Size(result); ++i) {
    PyObject *py_capsule = PyTuple_GetItem(result,i);
    DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(py_capsule, "dltensor");
    if (dlmanaged_tensor) {
      auto ort_value = onnxruntime::python::DlpackToOrtValue(dlmanaged_tensor);
      returned_args.push_back(ort_value);
      // We transfer our owning reference to the PyCapsule into the
      // ort_value (where it will be used in the eventual callback to
      // the deleter carried in the DLPack).  Hence we increment the
      // ref-count on py_capsule here in order to pre-compensate for
      // the decrement we will make when releasing our reference on
      // the result tuple.
      Py_INCREF(py_capsule);
    } else {
      // Clear error condition.  Is this case expected other than
      // returning None as the context object?
      ::std::cerr << "No value from dlpack at " << i << "\n";
      PyErr_Clear();
    }
  }
  Py_DECREF(result);
}

// Generate a tuple of arguments for use in a call, and return an
// owning refernece to it.
//
// We borrow references to the callback function, and (via PyObject*
// in obj_args) to the constant / non-tensor args.
  
PyObject* CreateForwardArguments(
    PyObject* callback,
    const size_t len,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<PyObject*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    bool is_training_mode) {
  ORT_ENFORCE(PyCallable_Check(callback), "Forward callback is not callable.");
  PyObject* args = PyTuple_New(4 + len);
  PyObject* tensor_flags = CreateTensorFlags(len, tensor_indices);
  PyObject* requires_grad_flags = CreateRequiresGradFlags(requires_grads);

  // Currently we are borrowing a reference to callback.  Hence we
  // need to increment the ref-count prior to storting into the tuple
  // of args, given that PyTuple_SetItem will steal a reference.
  Py_INCREF(callback);
  PyTuple_SetItem(args, 0, callback);
  PyTuple_SetItem(args, 1, requires_grad_flags);
  PyTuple_SetItem(args, 2, tensor_flags);
  PyTuple_SetItem(args, 3, is_training_mode ? Py_True : Py_False);

  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Wrap with DLPack, then transfer to Python for its release.  In
    // each case we allocate a new DLPack (getting an owning
    // reference) which we then allow to be stolen when passing in to
    // PyTuple_SetItem.
    DLManagedTensor* dlmanaged_tensor = onnxruntime::python::OrtValueToDlpack(*tensor_args[i]);
    PyObject* dltensor = PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
    PyTuple_SetItem(args, 4 + tensor_indices[i], dltensor);
  }

  for (size_t i = 0; i < obj_args.size(); ++i) {
    // Pass constant / non-tensor args into the tuple.  As above, we
    // are currently borrowing these, and so must increment the
    // ref-count to allow the tuple steal.  Otherwise we will free the
    // underlying constant / non-tensor arg when freeing the tuple.
    Py_INCREF(reinterpret_cast<PyObject*>(obj_args[i]));
    PyTuple_SetItem(args, 4 + obj_indices[i], reinterpret_cast<PyObject*>(obj_args[i]));
  }

  return args;
}

// Invoke the underlying forward/backward function.
//
// We borrow a reference to the runner and callback objects, and
// borrow references to the constant / non-tensor args passed in
// obj_args.
  
void Invoke(
    PyObject* runner,
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<PyObject*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<OrtValue>& returned_args,
    bool is_training_mode) {
  const auto len = tensor_args.size() + obj_args.size();
  CheckArguments(len, requires_grads, tensor_args, tensor_indices, obj_args, obj_indices);

  // Initialize arguments, and generate an owning reference to them
  PyObject* args = CreateForwardArguments(
      callback,
      len,
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      is_training_mode);

  InvokeRunner(runner, args, returned_args);
  
  // Decrement the owning reference on the arguments (in turn decrementing the counts on the items inside)
  Py_DECREF(args);
}

void TorchProxy::Forward(
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<PyObject*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<OrtValue>& returned_args,
    bool is_training_mode) {
  auto runner = OrtTorchFunctionPool::GetInstance().GetForwardRunner();
  Invoke(
      runner,
      callback,
      requires_grads,
      tensor_args,
      tensor_indices,
      obj_args,
      obj_indices,
      returned_args,
      is_training_mode);
}

void TorchProxy::Backward(
    PyObject* callback,
    const std::vector<int64_t>& requires_grads,
    const std::vector<OrtValue*>& tensor_args,
    const std::vector<int64_t>& tensor_indices,
    const std::vector<PyObject*>& obj_args,
    const std::vector<int64_t>& obj_indices,
    std::vector<OrtValue>& returned_args) {
  auto runner = OrtTorchFunctionPool::GetInstance().GetBackwardRunner();
  Invoke(
      runner,
      callback,
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
