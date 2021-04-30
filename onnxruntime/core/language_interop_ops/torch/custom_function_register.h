// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/env.h"
#include <mutex>
#include <Python.h>

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

class OrtTorchFunctionPool final {
 public:
  static OrtTorchFunctionPool& GetInstance() {
    static OrtTorchFunctionPool instance_;
    return instance_;
  }
  void RegisterForward(const std::string& custom_function_name, PyObject* forward_fn);
  void RegisterBackward(const std::string& custom_function_name, PyObject* backward_fn);
  void RegisterObject(PyObject* obj);
  void RegisterForwardCore(const std::string& key, PyObject* obj);
  void RegisterBackwardCore(const std::string& key, PyObject* obj);
  void RegisterForwardRunner(PyObject* obj);
  void RegisterBackwardRunner(PyObject* obj);
  PyObject* GetForwardRunner();
  PyObject* GetBackwardRunner();
  PyObject* GetForwardCore(const std::string& key);
  PyObject* GetBackwardCore(const std::string& key);
  PyObject* GetForward(const std::string& custom_function_name);
  PyObject* GetBackward(const std::string& custom_function_name);

  int64_t RegisterContext(PyObject* auto_grad_context);

  PyObject* GetContext(int64_t context_index);
  void UnRegisterContext(int64_t context_index);

 private:
  OrtTorchFunctionPool() = default;
  ~OrtTorchFunctionPool() = default;
  OrtTorchFunctionPool(const OrtTorchFunctionPool&) = delete;

  OrtTorchFunctionPool& operator=(const OrtTorchFunctionPool&) = delete;

  std::vector<PyObject*> obj_pool;
  PyObject* forward_runner;
  PyObject* backward_runner;
  std::unordered_map<std::string, PyObject*> forward_core_pool;
  std::unordered_map<std::string, PyObject*> backward_core_pool;

  std::unordered_map<std::string, PyObject*> forward_pool;
  std::unordered_map<std::string, PyObject*> backward_pool;

  std::unordered_map<int64_t, PyObject*> func_context_pool;
  std::mutex func_context_pool_mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
