// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <utility>
#include "core/platform/env.h"

namespace onnxruntime {

/**
  * Unloads dynamic library handles loaded via Env::LoadDynamicLibrary() upon destruction.
  */
struct LibraryHandles {
  LibraryHandles() = default;
  ~LibraryHandles();

  LibraryHandles(const LibraryHandles&) = delete;
  LibraryHandles& operator=(const LibraryHandles&) = delete;

  // Move-only.
  LibraryHandles(LibraryHandles&& other);
  LibraryHandles& operator=(LibraryHandles&& other);

  // Add a library handle that should be unloaded upon destruction of this object.
  // The `library_name` is used for error reporting if unloading should fail.
  void Add(std::string library_name, void* library_handle);

 private:
  std::vector<std::pair<std::string, void*>> libraries_;

  void UnloadLibraries() noexcept;
};
}  // namespace onnxruntime