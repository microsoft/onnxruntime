// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/env.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_context_internal.h"
#include <Python.h>
namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

using OnnxAttrs = std::unordered_map<std::string, std::string>;

class TorchProxy {
 public:
  static TorchProxy& GetInstance();

  void Forward(
      PyObject* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<PyObject*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<OrtValue>& outputs,
      bool is_training_mode);

  void Backward(
      PyObject* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<PyObject*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<OrtValue>& outputs);

  bool Initialized() const { return initialized_; };
  int32_t GetGil() const;
  void PutGil(int32_t) const;

 private:
  TorchProxy();
  ~TorchProxy();
  bool initialized_ = false;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
