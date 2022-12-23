// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <utility>
#include "core/platform/env.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {

/**
  * Class that owns a collection of dynamic library handles and unloads the library handles when the class instance
  * goes out of scope.
  *
  * Use LibraryHandles::Add() to add a dynamic library handle to an instance of this class.
  * The destructor unloads all added library handles via Env::UnloadDynamicLibrary().
  *
  * This class is currently used in SessionOptions to manage the lifetime of custom operator library handles that have
  * been registered via OrtApi::RegisterCustomOpsLibrary_V2.
  */
struct LibraryHandles {
  LibraryHandles() = default;
  ~LibraryHandles();

  // Move-only.
  LibraryHandles(LibraryHandles&& other);
  LibraryHandles& operator=(LibraryHandles&& other);

  // Add a library handle that should be unloaded upon destruction of this object.
  // The `library_name` is used for error reporting if unloading should fail.
  void Add(std::string library_name, void* library_handle);

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LibraryHandles);

  void UnloadLibraries() noexcept;

  InlinedVector<std::pair<std::string, void*>> libraries_;
};
}  // namespace onnxruntime