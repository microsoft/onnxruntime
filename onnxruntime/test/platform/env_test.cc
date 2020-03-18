// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"

#include <fstream>

#include "gtest/gtest.h"

#include "core/common/path_string.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(PlatformEnvTest, DirectoryCreationAndDeletion) {
  const auto& env = Env::Default();
  const PathString root_dir = ORT_TSTR("tmp_platform_env_test_dir");
  const PathString sub_dir = root_dir + ORT_TSTR("/some/test/directory");

  ASSERT_FALSE(env.FolderExists(root_dir));

  ASSERT_STATUS_OK(env.CreateFolder(sub_dir));
  ASSERT_TRUE(env.FolderExists(sub_dir));

  // create a file in the subdirectory
  {
    std::ofstream outfile{sub_dir + ORT_TSTR("/file")};
    outfile << "hello!";
  }

  ASSERT_STATUS_OK(env.DeleteFolder(root_dir));
  ASSERT_FALSE(env.FolderExists(root_dir));
}

}  // namespace test
}  // namespace onnxruntime
