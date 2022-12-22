// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/library_handle.h"
#include "core/common/logging/logging.h"
#include <utility>

namespace onnxruntime {
LibraryHandle::LibraryHandle(void* raw_handle, std::string lib_name, const Env& env) : raw_handle_(raw_handle),
                                                                                       lib_name_(std::move(lib_name)),
                                                                                       env_(env) {}
LibraryHandle::~LibraryHandle() {
  if (raw_handle_ != nullptr) {
    auto status = env_.UnloadDynamicLibrary(raw_handle_);
    if (!status.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to unload handle for dynamic library " << lib_name_;
    }
  }
}

LibraryHandle::LibraryHandle(LibraryHandle&& other) : raw_handle_(std::exchange(other.raw_handle_, nullptr)),
                                                      lib_name_(std::move(other.lib_name_)),
                                                      env_(other.env_) {}

}  // namespace onnxruntime