// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef USE_CUDA_MINIMAL

#include <mutex>
#include <string>
#include <unordered_map>

#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime::cuda {

// Lazily loads the cuFFT runtime library and resolves its symbols on demand so
// that the CUDA Execution Provider does not carry a hard link-time dependency on
// cuFFT. cuFFT is only required by the FFT contrib ops (Rfft/Irfft); when the
// library is not present those ops fail with NOT_IMPLEMENTED while the rest of
// the provider keeps working.
class CufftLibrary {
 public:
  static CufftLibrary& Get();

  bool Available();
  std::string Error() const;

  template <typename T>
  T Resolve(const char* symbol) {
    return reinterpret_cast<T>(ResolveSymbol(symbol));
  }

 private:
  CufftLibrary() = default;

  bool EnsureLoaded();
  void* ResolveSymbol(const char* symbol);

  mutable std::mutex mutex_;
  bool load_attempted_{false};
  bool available_{false};
  std::string error_;
  void* handle_{nullptr};
  std::unordered_map<std::string, void*> symbols_;
};

const char* CufftUnavailableErrorString();

}  // namespace onnxruntime::cuda

#endif  // USE_CUDA_MINIMAL
