// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <unordered_map>
#include <mutex>
#include <Python.h>

namespace onnxruntime {
namespace python {

class OrtTorchFunctionPool final {
 public:
  static OrtTorchFunctionPool& GetInstance() {
    static OrtTorchFunctionPool instance_;
    return instance_;
  }
  void RegisterForward(const std::string& custom_function_name, PyObject* forward_fn);
  void RegisterBackward(const std::string& custom_function_name, PyObject* backward_fn);
  PyObject* GetForward(const std::string& custom_function_name);
  PyObject* GetBackward(const std::string& custom_function_name);

  int64_t RegisterContext(PyObject* auto_grad_context);

  void UnRegisterContext(int64_t context_index);

 private:
  OrtTorchFunctionPool() = default;
  ~OrtTorchFunctionPool() = default;
  OrtTorchFunctionPool(const OrtTorchFunctionPool&) = delete;

  OrtTorchFunctionPool& operator=(const OrtTorchFunctionPool&) = delete;

  std::unordered_map<std::string, PyObject*> forward_pool;
  std::unordered_map<std::string, PyObject*> backward_pool;

  std::unordered_map<int64_t, PyObject*> func_context_pool;
  std::mutex func_context_pool_mutex_;
};
}  // namespace python
}  // namespace onnxruntime
