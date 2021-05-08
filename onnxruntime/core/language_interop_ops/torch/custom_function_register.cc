// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Python.h>
#include <mutex>
#include "core/common/common.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

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
  Py_INCREF(forward_fn);
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
  Py_INCREF(backward_fn);
  backward_pool[custom_function_name] = backward_fn;
}

void OrtTorchFunctionPool::RegisterObject(PyObject* obj) {
  Py_INCREF(obj);
  obj_pool.push_back(obj);
}

void OrtTorchFunctionPool::RegisterForwardRunner(PyObject* obj) {
  Py_INCREF(obj);
  forward_runner = obj;
}
void OrtTorchFunctionPool::RegisterBackwardRunner(PyObject* obj) {
  Py_INCREF(obj);
  backward_runner = obj;
}

PyObject* OrtTorchFunctionPool::GetForwardRunner() {
  return forward_runner;
}
PyObject* OrtTorchFunctionPool::GetBackwardRunner() {
  return backward_runner;
}

void OrtTorchFunctionPool::RegisterForwardCore(const std::string& key, PyObject* obj) {
  Py_INCREF(obj);
  forward_core_pool[key] = obj;
}

void OrtTorchFunctionPool::RegisterBackwardCore(const std::string& key, PyObject* obj) {
  Py_INCREF(obj);
  backward_core_pool[key] = obj;
}

PyObject* OrtTorchFunctionPool::GetForwardCore(const std::string& key) {
  return forward_core_pool.at(key);
}

PyObject* OrtTorchFunctionPool::GetBackwardCore(const std::string& key) {
  return backward_core_pool.at(key);
}

PyObject* OrtTorchFunctionPool::GetForward(
    const std::string& custom_function_name) {
  auto it = forward_pool.find(custom_function_name);
  ORT_ENFORCE(it != forward_pool.end(), "No custom forward function registered for ", custom_function_name);
  return it->second;
}

PyObject* OrtTorchFunctionPool::GetBackward(
    const std::string& custom_function_name) {
  auto it = backward_pool.find(custom_function_name);
  ORT_ENFORCE(it != backward_pool.end(), "No custom backward function registered for ", custom_function_name);
  return it->second;
}

int64_t OrtTorchFunctionPool::RegisterContext(PyObject* auto_grad_context) {
  static int64_t index_ = 0;
  std::unique_lock<std::mutex> lk(func_context_pool_mutex_);
  index_++;
  PyObject_Print(auto_grad_context, stdout, 0);
  func_context_pool.insert({index_, auto_grad_context});
  Py_INCREF(auto_grad_context);
  return index_;
}

PyObject* OrtTorchFunctionPool::GetContext(int64_t context_index) {
  auto ctx = func_context_pool.find(context_index);
  return ctx->second;
}

void OrtTorchFunctionPool::UnRegisterContext(int64_t context_index) {
  std::unique_lock<std::mutex> lk(func_context_pool_mutex_);
  auto ctx = func_context_pool.find(context_index);
  PyObject_Print(ctx->second, stdout, 0);
  Py_XDECREF(ctx->second);
  func_context_pool.erase(ctx);
}

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
