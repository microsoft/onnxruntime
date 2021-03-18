// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <unordered_map>
#include <pybind11/pybind11.h>

namespace onnxruntime {
namespace python {

class OrtTorchFunctionPool final {
 public:
  static OrtTorchFunctionPool& GetInstance() {
      static OrtTorchFunctionPool instance_;
      return instance_;
  }
  void RegisterForward(const std::string& custom_function_name, pybind11::object forward_fn);
  void RegisterBackward(const std::string& custom_function_name, pybind11::object backward_fn);
  pybind11::object GetForward(const std::string& custom_function_name);
  pybind11::object GetBackward(const std::string& custom_function_name);

 private:
  OrtTorchFunctionPool() = default;
  ~OrtTorchFunctionPool() = default;
  OrtTorchFunctionPool(const OrtTorchFunctionPool&) = delete;

  OrtTorchFunctionPool& operator=(const OrtTorchFunctionPool&) = delete;

  std::unordered_map<std::string, pybind11::object> forward_pool;
  std::unordered_map<std::string, pybind11::object> backward_pool;
};
} // namespace python
} // namespace onnxruntime
