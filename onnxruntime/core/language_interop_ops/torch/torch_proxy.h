// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/env.h"
#define LOAD_PYOP_SYM(n, v, m) ORT_ENFORCE(Env::Default().GetSymbolFromLibrary(handle_, n, reinterpret_cast<void**>(&v)) == Status::OK(), m)
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#ifdef _WIN32
#include <Windows.h>
#else
#define HMODULE void*
#endif

namespace onnxruntime {
using OnnxAttrs = std::unordered_map<std::string, std::string>;

class TorchProxy {
 public:
  static TorchProxy& GetInstance();

  void Forward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<void*>& outputs);

  void Backward(
      void* callback,
      const std::vector<int64_t>& requires_grads,
      const std::vector<OrtValue*>& tensor_args,
      const std::vector<int64_t>& tensor_indices,
      std::vector<void*>& obj_args,
      const std::vector<int64_t>& obj_indices,
      std::vector<void*>& outputs);

  bool Initialized() const { return initialized_; };
  int32_t GetGil() const;
  void PutGil(int32_t) const;

 private:
  TorchProxy();
  ~TorchProxy();
  bool initialized_ = false;
};

}  // namespace onnxruntime
