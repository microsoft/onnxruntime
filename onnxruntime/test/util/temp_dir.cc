// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/temp_dir.h"

#include "gtest/gtest.h"

#include "core/platform/env.h"
#include "test/util/include/asserts.h"
namespace onnxruntime {
namespace test {

namespace {
void DeleteDirectory(const PathString& path) {
  ASSERT_STATUS_OK(Env::Default().DeleteFolder(path));
}

void CreateDirectory(const PathString& path, bool delete_if_exists) {
  const bool exists = Env::Default().FolderExists(path);
  if (exists) {
    if (delete_if_exists) {
      DeleteDirectory(path);
    } else {
      FAIL() << "Temporary directory " << path << " already exists.";
    }
  }

  ASSERT_STATUS_OK(Env::Default().CreateFolder(path));
}

}  // namespace

TemporaryDirectory::TemporaryDirectory(const PathString& path, bool delete_if_exists)
    : path_{path} {
  CreateDirectory(path, delete_if_exists);
}

TemporaryDirectory::~TemporaryDirectory() {
  DeleteDirectory(path_);
}

}  // namespace test
}  // namespace onnxruntime
