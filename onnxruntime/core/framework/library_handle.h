// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/platform/env.h"

namespace onnxruntime {

struct LibraryHandle {
  LibraryHandle(void* raw_handle, std::string lib_name, const Env& env);
  ~LibraryHandle();

  LibraryHandle(const LibraryHandle&) = delete;
  LibraryHandle& operator=(const LibraryHandle&) = delete;

  LibraryHandle(LibraryHandle&& other);
  LibraryHandle& operator=(LibraryHandle&& other) = delete;

 private:
  void* raw_handle_;
  std::string lib_name_;
  const Env& env_;
};
}  // namespace onnxruntime