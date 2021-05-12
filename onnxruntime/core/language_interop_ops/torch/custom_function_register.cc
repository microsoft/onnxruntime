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

void OrtTorchFunctionPool::RegisterObject(PyObject* obj) {
  ORT_ENFORCE(obj, "Cannot register NULL.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  Py_INCREF(obj);
  obj_pool.push_back(obj);
}

void OrtTorchFunctionPool::RegisterForwardRunner(PyObject* obj) {
  ORT_ENFORCE(obj, "Cannot register NULL.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  Py_INCREF(obj);
  forward_runner = obj;
}

void OrtTorchFunctionPool::RegisterBackwardRunner(PyObject* obj) {
  ORT_ENFORCE(obj, "Cannot register NULL.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  Py_INCREF(obj);
  backward_runner = obj;
}

PyObject* OrtTorchFunctionPool::GetForwardRunner() {
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  ORT_ENFORCE(forward_runner, "Forward runner cannot be NULL. Do you forget register it by calling RegisterForwardRunner(...)?");
  return forward_runner;
}

PyObject* OrtTorchFunctionPool::GetBackwardRunner() {
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  ORT_ENFORCE(backward_runner, "backward runner cannot be NULL. Do you forget register it by calling RegisterBackwardRunner(...)?");
  return backward_runner;
}

// The input argument "obj" should be "apply" of
// class CustomFunction(autograd.Function):
//   @staticmethod
//   def forward(ctx, ...):
//     ...
//   @staticmethod
//   def backward(ctx, ...):
//     ...
// That is,
//   obj = CustomFunction.apply
void OrtTorchFunctionPool::RegisterForwardCore(const std::string& key, PyObject* obj) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  ORT_ENFORCE(obj, "Cannot register NULL.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  Py_INCREF(obj);
  forward_core_pool[key] = obj;
}

// The input argument "obj" should be "backward" of
// class CustomFunction(autograd.Function):
//   @staticmethod
//   def forward(ctx, ...):
//     ...
//   @staticmethod
//   def backward(ctx, ...):
//     ...
// That is,
//   obj = CustomFunction.backward
void OrtTorchFunctionPool::RegisterBackwardCore(const std::string& key, PyObject* obj) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  ORT_ENFORCE(obj, "Cannot register NULL.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  Py_INCREF(obj);
  backward_core_pool[key] = obj;
}

// The "key" is the "name" attribute in PythonOp.
PyObject* OrtTorchFunctionPool::GetForwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  auto iter = forward_core_pool.find(key);
  ORT_ENFORCE(iter != forward_core_pool.end(), "No forward registered for ", key);
  return iter->second;
}

// The "key" is the "name" attribute in PythonOp.
PyObject* OrtTorchFunctionPool::GetBackwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  auto iter = backward_core_pool.find(key);
  ORT_ENFORCE(iter != backward_core_pool.end(), "No backward registered for ", key);
  return iter->second;
}

int64_t OrtTorchFunctionPool::RegisterContext(PyObject* auto_grad_context) {
  static int64_t index_ = 0;
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  index_++;
  PyObject_Print(auto_grad_context, stdout, 0);
  func_context_pool.insert({index_, auto_grad_context});
  Py_INCREF(auto_grad_context);
  return index_;
}

PyObject* OrtTorchFunctionPool::GetContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  auto iter = func_context_pool.find(context_index);
  ORT_ENFORCE(iter != func_context_pool.end(), "No context registered for ", context_index);
  return iter->second;
}

void OrtTorchFunctionPool::UnRegisterContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(func_context_pool_mutex_);
  auto iter = func_context_pool.find(context_index);
  ORT_ENFORCE(iter != func_context_pool.end(), "No context registered for ", context_index);
  PyObject_Print(iter->second, stdout, 0);
  Py_XDECREF(iter->second);
  func_context_pool.erase(iter);
}

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
