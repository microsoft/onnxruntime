// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/library_handles.h"
#include "core/common/logging/logging.h"
#include <utility>

namespace onnxruntime {
LibraryHandles::~LibraryHandles() noexcept {
  UnloadLibraries();
}

LibraryHandles::LibraryHandles(LibraryHandles&& other) noexcept : libraries_(std::move(other.libraries_)) {}

LibraryHandles& LibraryHandles::operator=(LibraryHandles&& other) noexcept {
  if (this != &other) {
    UnloadLibraries();

    libraries_ = std::move(other.libraries_);
  }

  return *this;
}

void LibraryHandles::Add(PathString library_name, void* library_handle) {
  libraries_.push_back(std::make_pair(std::move(library_name), library_handle));
}

void LibraryHandles::UnloadLibraries() noexcept {
  if (!libraries_.empty()) {
    const Env& env = Env::Default();

    for (const auto& it : libraries_) {
      auto status = env.UnloadDynamicLibrary(it.second);
      if (!status.IsOK()) {
        LOGS_DEFAULT(WARNING) << "Failed to unload handle for dynamic library "
                              << onnxruntime::PathToUTF8String(it.first) << ": " << status;
      }
    }
  }
}

}  // namespace onnxruntime
