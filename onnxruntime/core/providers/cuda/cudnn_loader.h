// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef USE_CUDA_MINIMAL

#include <mutex>
#include <string>
#include <unordered_map>

#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime::cuda {

class CudnnLibrary {
 public:
  static CudnnLibrary& Get();

  bool Available();
  std::string Error() const;
  void* Handle();

  template <typename T>
  T Resolve(const char* symbol) {
    return reinterpret_cast<T>(ResolveSymbol(symbol));
  }

 private:
  CudnnLibrary() = default;

  bool EnsureLoaded();
  void* ResolveSymbol(const char* symbol);

  mutable std::mutex mutex_;
  bool load_attempted_{false};
  bool available_{false};
  std::string error_;
  void* handle_{nullptr};
  std::unordered_map<std::string, void*> symbols_;
};

const char* CudnnUnavailableErrorString();

}  // namespace onnxruntime::cuda

#endif  // USE_CUDA_MINIMAL
