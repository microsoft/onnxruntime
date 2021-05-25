// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <Python.h>
#include <mutex>

#ifndef SHARED_PROVIDER
#include "core/platform/env.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_context_internal.h"
#endif

#include "core/language_interop_ops/torch/object_pointer.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// PyObject RAII wrapper
using PythonObjectPtr = ObjectPointer<PyObject>;
template class ObjectPointer<PyObject>;

class TorchProxy {
 public:
  static TorchProxy& GetInstance() {
    static TorchProxy proxy;
    return proxy;
  };

  void Forward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      void** diff_ctx,
      std::vector<OrtValue>& returned_ortvalues,
      const bool is_training_mode,
      const bool is_inplace);

  void Backward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<OrtValue>& return_args,
      const bool is_inplace);

 private:
  TorchProxy(){};
  ~TorchProxy(){};
  TorchProxy(const TorchProxy&) = delete;
  TorchProxy& operator=(const TorchProxy&) = delete;
  // All member functions should be exclusively used because
  // Python has a global interpreter.
  std::mutex mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
