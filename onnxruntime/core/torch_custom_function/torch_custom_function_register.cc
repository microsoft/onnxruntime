// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "torch_custom_function_register.h"
#include <Python.h>
#include <mutex>

namespace onnxruntime {
namespace python {

void OrtTorchFunctionPool::RegisterForward(
    const std::string& custom_function_name,
    PyObject* forward_fn) {
  // This should be "apply" of
  // class custom_function_name(autograd.Function):
  //   @staticmethod
  //   def forward(ctx, ...):
  //     ...
  //   @staticmethod
  //   def backward(ctx, ...):
  //     ...
  // That is,
  //   forward_fn = custom_function_name.apply
  forward_pool[custom_function_name] = forward_fn;
}

void OrtTorchFunctionPool::RegisterBackward(
    const std::string& custom_function_name,
    PyObject* backward_fn) {
  // This should be "backward" of
  // class custom_function_name(autograd.Function):
  //   @staticmethod
  //   def forward(ctx, ...):
  //     ...
  //   @staticmethod
  //   def backward(ctx, ...):
  //     ...
  // That is,
  //   backward_fn = custom_function_name.backward
  backward_pool[custom_function_name] = backward_fn;
}

PyObject* OrtTorchFunctionPool::GetForward(
    const std::string& custom_function_name) {
  return forward_pool.at(custom_function_name);
};

PyObject* OrtTorchFunctionPool::GetBackward(
    const std::string& custom_function_name) {
  return backward_pool.at(custom_function_name);
};

int64_t OrtTorchFunctionPool::RegisterContext(PyObject* auto_grad_context) {
  static int64_t index_ = 0;
  std::unique_lock<std::mutex> lk(func_context_pool_mutex_);
  index_++;
  func_context_pool.insert({index_, auto_grad_context});
  Py_INCREF(auto_grad_context);
  return index_;
};

void OrtTorchFunctionPool::UnRegisterContext(int64_t context_index) {
  std::unique_lock<std::mutex> lk(func_context_pool_mutex_);
  auto ctx = func_context_pool.find(context_index);
  Py_XDECREF(ctx->second);
  func_context_pool.erase(ctx);
};

}  // namespace python
}  // namespace onnxruntime
