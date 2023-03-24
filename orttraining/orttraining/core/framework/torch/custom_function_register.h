// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "orttraining/core/framework/torch/python_common.h"

#include "orttraining/core/framework/torch/torch_proxy.h"
#include <mutex>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

class OrtTorchFunctionPool final {
 public:
  static OrtTorchFunctionPool& GetInstance() {
    static OrtTorchFunctionPool instance_;
    return instance_;
  }

  // AutogradFunction includes ForwardCore and BackwardCore.
  // ForwardCore is the apply() function pointer.
  // BackwardCore is the backward() function pointer.
  // RegisterTorchAutogradFunction owns the input "obj" and will release its ownership only in its destructor.
  void RegisterTorchAutogradFunction(const std::string& key, PyObject* obj);
  // Return a borrowed reference to the stored Python function. Thus,
  //  1. The returned value doesn't own its Python function.
  //  2. Caller of GetForwardCore should not decrease the reference count of the returned object.
  PyObject* GetForwardCore(const std::string& key);  // The "key" is the "name" attribute in PythonOp.
  // Return a borrowed reference to the stored Python function. Thus,
  //  1. The returned value doesn't own its Python function.
  //  2. Caller of GetBackwardCore should not decrease the reference count of the returned object.
  PyObject* GetBackwardCore(const std::string& key);  // The "key" is the "name" attribute in PythonOpGrad.

  // Autograd function may take input of "non-tensor && non int/float && non int/float tuple" types.
  // While PythonOp running requires those inputs be there otherwise kernel execution will fail.
  // So during model exporting, we need register those input with this API, then a ref cnt is increased by 1,
  // they will not be released until OrtTorchFunctionPool is destroyed.
  // We also trying to release those registration in 'UnRegisterFunctions' to avoid the issues of python program
  // exits before we de-crease ref cnt for the already release python object.
  void RegisterMiscellaneousConstInput(PyObject* obj);

  // Context is torch backward gradient function pointer, and
  // it is a property of forward run outputs (tensors), its lifecycle
  // is along with forward run outputs in PyTorch design.
  // Register a borrowed Python object in forward pass.
  int64_t RegisterContext(PyObject* auto_grad_context);
  // Unregister a borrowed Python object in backward pass.
  // It doesn't decrease reference count of the underlying Python object
  // but remove the index-context pair from "func_context_pool_".
  void UnregisterContext(int64_t index);
  // Retrieve the context associated with the index.
  PyObject* GetContext(int64_t index);

  // ForwardRunner/BackwardRunner are "glue" codes written in Python that interacting
  // with C++ kernels during Python function invoking.
  // This function creates new ownership to "obj".
  void RegisterForwardRunner(PyObject* obj);
  // This function creates new ownership to "obj".
  void RegisterBackwardRunner(PyObject* obj);
  // Return a borrowed reference to a Python function, which
  // is responsible for executing autograd.Function.apply.
  PyObject* GetForwardRunner();
  // Return a borrowed reference to a Python function, which
  // is responsible for executing autograd.Function.apply.
  PyObject* GetBackwardRunner();

  // The reason we provide this unregister api is:
  //   A static OrtTorchFunctionPool instance will be destructed after
  //   Python modules/functions are released. Once we own func pointers
  //   by increasing ref count for the functions, we need decrease the
  //   ref count in ~OrtTorchFunctionPool, but at that time some properties
  //   of the python function object, for example co_consts
  //   (tuple type, https://github.com/python/cpython/blob/3.7/Objects/funcobject.c#L38)
  //   already released, there will be a segment fault.

  //   Calling this function upon normal interpreter termination helps release the
  //   registered functions earlier, avoiding segment fault.
  void UnRegisterFunctions();

 private:
  OrtTorchFunctionPool(){};
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OrtTorchFunctionPool);

  void UnRegisterGlobalFunctions();
  void UnRegisterModelSpecificFunctions();

  PythonObjectPtr forward_runner_;
  PythonObjectPtr backward_runner_;

  std::unordered_map<std::string, PythonObjectPtr> forward_core_pool_;
  std::unordered_map<std::string, PythonObjectPtr> backward_core_pool_;
  std::unordered_map<std::string, PythonObjectPtr> miscellaneous_const_input_pool_;
  std::unordered_map<int64_t, PythonObjectPtr> func_context_pool_;

  std::mutex mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
