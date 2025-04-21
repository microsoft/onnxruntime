// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct Provider;

struct ProviderLibrary {
  ProviderLibrary(const ORTCHAR_T* filename, bool unload = true);
  ~ProviderLibrary();

  Status Load();
  Provider& Get();
  void Unload();

 private:
  std::mutex mutex_;
  const ORTCHAR_T* filename_;
  bool unload_;
  bool initialized_{};
  Provider* provider_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderLibrary);
};
}  // namespace onnxruntime
