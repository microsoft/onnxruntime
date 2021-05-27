// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <mutex>

#ifndef SHARED_PROVIDER
#include "core/platform/env.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_context_internal.h"
#endif

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

/// Use void* instead of PyObject* to avoid add unnecessary
/// python.h dependency for the consumers.
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

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TorchProxy);

  // All member functions should be exclusively used because
  // Python has a global interpreter.
  std::mutex mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
