// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <stdio.h>
#include <string>

#include "core/common/path_string.h"

namespace onnxruntime {
namespace test {
void CreateTestFile(FILE*& out, std::basic_string<ORTCHAR_T>& filename_template);
void CreateTestFile(int& out, std::basic_string<ORTCHAR_T>& filename_template);
void DeleteFileFromDisk(const ORTCHAR_T* path);

class ScopedFileDeleter {
 public:
  ScopedFileDeleter() = default;
  ScopedFileDeleter(const PathString& path) : path_{path} {}
  ScopedFileDeleter(ScopedFileDeleter&& other) noexcept { *this = std::move(other); }
  ScopedFileDeleter& operator=(ScopedFileDeleter&& other) noexcept {
    CleanUp();
    path_ = std::move(other.path_);
    other.path_.clear();
    return *this;
  }
  ~ScopedFileDeleter() { CleanUp(); }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ScopedFileDeleter);

  void CleanUp() {
    if (!path_.empty()) {
      std::remove(ToMBString(path_).c_str());
      path_.clear();
    }
  }

  PathString path_;
};

}  // namespace test
}  // namespace onnxruntime