// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/env.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_context_internal.h"
namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

class TorchProxy {
 public:
  static TorchProxy& GetInstance();

  void Forward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<void*>& outputs,
      bool is_training_mode);

  void Backward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      const std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<void*>& outputs);

 private:
  TorchProxy();
  ~TorchProxy();
  TorchProxy(const TorchProxy&) = delete;
  TorchProxy& operator=(const TorchProxy&) = delete;

  bool initialized_ = false;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
