// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/temp_dir.h"

#include "gtest/gtest.h"

#include "core/platform/env.h"

namespace onnxruntime {
namespace test {
namespace {
void CreateOrDeleteDirectory(const PathString& path, bool create, bool throw_on_fail = true) {
  const auto status = create ? Env::Default().CreateFolder(path) : Env::Default().DeleteFolder(path);
  EXPECT_TRUE(status.IsOK()) << "Failed to " << (create ? "create" : "delete") << "temporary directory " << path;

  if (throw_on_fail) {
    ORT_ENFORCE(status.IsOK());
  }
}
}  // namespace

TemporaryDirectory::TemporaryDirectory(const PathString& path, bool delete_if_exists)
    : path_{path} {
  // EXPECT and throw to fail even if anyone is catching exceptions
  const bool exists = Env::Default().FolderExists(path_);
  if (exists) {
    if (!delete_if_exists) {
      EXPECT_FALSE(exists) << "Temporary directory " << path_ << " already exists.";
      ORT_ENFORCE(!exists);
    }

    CreateOrDeleteDirectory(path_, /* create */ false);
  }

  CreateOrDeleteDirectory(path_, /* create*/ true);
}

TemporaryDirectory::~TemporaryDirectory() {
  // don't throw in dtor
  CreateOrDeleteDirectory(path_, /* create */ false, /* throw_on_fail */ false);
}

}  // namespace test
}  // namespace onnxruntime
