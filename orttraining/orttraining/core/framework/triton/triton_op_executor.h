// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "orttraining/core/framework/torch/python_common.h"

#ifndef SHARED_PROVIDER
#include "core/framework/ort_value.h"
#endif

namespace onnxruntime {
namespace training {
namespace framework {
namespace triton {

using PythonObjectPtr = std::unique_ptr<PyObject, std::function<void(PyObject*)>>;

class TritonOpExecutor final {
 public:
  static TritonOpExecutor& Instance() {
    static TritonOpExecutor instance;
    return instance;
  }

  void Initialize(PyObject* config_getter, PyObject* executor_by_name, PyObject* executor_by_onnx);

  bool IsInitialized() {
    return config_getter_.get() != nullptr && executor_by_name_.get() != nullptr && executor_by_onnx_.get() != nullptr;
  }

  std::string GetConfigJson();

  // Execute ONNX graph by codegening, compiling and executing Triton kernels.
  void ExecuteByOnnx(int64_t onnx_key, const std::string& onnx_string, const InlinedVector<const OrtValue*>& inputs,
                     InlinedVector<OrtValue>& outputs, const InlinedHashSet<size_t>& bool_outputs = {});

  // Execute existing Triton kernel by Python function name.
  void ExecuteByFuncName(const std::string& func_name, const InlinedVector<const OrtValue*>& inputs,
                         InlinedVector<OrtValue>& outputs, const InlinedHashSet<size_t>& bool_outputs = {},
                         const InlinedHashMap<std::string, std::pair<std::string, int>>& kwargs = {});

 private:
  PythonObjectPtr config_getter_;
  PythonObjectPtr executor_by_name_;
  PythonObjectPtr executor_by_onnx_;
};

}  // namespace triton
}  // namespace framework
}  // namespace training
}  // namespace onnxruntime
