// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITONOP

#pragma once

#include "orttraining/core/framework/torch/torch_proxy.h"

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

  void Initialize(PyObject* config_getter, PyObject* executor) {
    ORT_ENFORCE(config_getter_.get() == nullptr && executor_.get() == nullptr && config_getter != nullptr &&
                executor != nullptr);
    Py_INCREF(config_getter);
    Py_INCREF(executor);
    PythonObjectPtr config_getter_ptr(config_getter, PythonObjectDeleter);
    config_getter_ = std::move(config_getter_ptr);
    PythonObjectPtr executor_ptr(executor, PythonObjectDeleter);
    executor_ = std::move(executor_ptr);
  }

  bool IsInitialized() { return config_getter_.get() != nullptr && executor_.get() != nullptr; }

  std::string GetConfigJson() {
    ORT_ENFORCE(config_getter_.get() != nullptr);
    PythonObjectPtr ret(PyObject_CallObject(config_getter_.get(), nullptr), PythonObjectDeleter);
    char* buffer = nullptr;
    Py_ssize_t length;
    buffer = const_cast<char*>(PyUnicode_AsUTF8AndSize(ret.get(), &length));
    return std::string(buffer, length);
  }

  PyObject* GetExecutor() {
    ORT_ENFORCE(executor_.get() != nullptr);
    return executor_.get();
  }

 private:
  PythonObjectPtr config_getter_;
  PythonObjectPtr executor_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ENABLE_TRITONOP
