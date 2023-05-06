// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#pragma once

#include "orttraining/core/framework/torch/torch_proxy.h"
#include "orttraining/core/framework/torch/gil.h"

namespace onnxruntime {
namespace contrib {

using PythonObjectPtr = language_interop_ops::torch::PythonObjectPtr;
constexpr auto PythonObjectDeleter = language_interop_ops::torch::PythonObjectDeleter;

class TritonOpExecutor final {
 public:
  static TritonOpExecutor& Instance() {
    static TritonOpExecutor instance;
    return instance;
  }

  void Initialize(PyObject* config_getter, PyObject* executor_by_name, PyObject* executor_by_onnx) {
    ORT_ENFORCE(config_getter_.get() == nullptr && config_getter != nullptr && executor_by_name_.get() == nullptr &&
                executor_by_name != nullptr && executor_by_onnx_.get() == nullptr && executor_by_onnx != nullptr);
    Py_INCREF(config_getter);
    Py_INCREF(executor_by_name);
    Py_INCREF(executor_by_onnx);
    PythonObjectPtr config_getter_ptr(config_getter, PythonObjectDeleter);
    config_getter_ = std::move(config_getter_ptr);
    PythonObjectPtr executor_by_name_ptr(executor_by_name, PythonObjectDeleter);
    executor_by_name_ = std::move(executor_by_name_ptr);
    PythonObjectPtr executor_by_onnx_ptr(executor_by_onnx, PythonObjectDeleter);
    executor_by_onnx_ = std::move(executor_by_onnx_ptr);
  }

  bool IsInitialized() {
    return config_getter_.get() != nullptr && executor_by_name_.get() != nullptr && executor_by_onnx_.get() != nullptr;
  }

  std::string GetConfigJson() {
    // Python-related calls should happen only if guard is alive.
    GilGuard guard;
    ORT_ENFORCE(config_getter_.get() != nullptr);
    PythonObjectPtr ret(PyObject_CallObject(config_getter_.get(), nullptr), PythonObjectDeleter);
    char* buffer = nullptr;
    Py_ssize_t length;
    buffer = const_cast<char*>(PyUnicode_AsUTF8AndSize(ret.get(), &length));
    return std::string(buffer, length);
  }

  PyObject* GetExecutorByName() {
    ORT_ENFORCE(executor_by_name_.get() != nullptr);
    return executor_by_name_.get();
  }

  PyObject* GetExecutorByOnnx() {
    ORT_ENFORCE(executor_by_onnx_.get() != nullptr);
    return executor_by_onnx_.get();
  }

 private:
  PythonObjectPtr config_getter_;
  PythonObjectPtr executor_by_name_;
  PythonObjectPtr executor_by_onnx_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ENABLE_TRITON
