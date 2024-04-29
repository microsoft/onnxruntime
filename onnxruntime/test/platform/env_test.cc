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

TEST(PlatformEnvTest, GetErrnoInfo) {
  // command that should generate an errno error
  std::ifstream file("non_existent_file");
  ASSERT_TRUE(file.fail());
  auto [err, msg] = GetErrnoInfo();
  ASSERT_EQ(err, ENOENT);

#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

  // GetErrnoInfo uses strerror_r or strerror_s depending on the platform. use the unsafe std::sterror to get the
  // expected value given this is a unit test so doesn't have to be as robust.
  ASSERT_EQ(msg, std::strerror(ENOENT));

#if defined(_WIN32)
#pragma warning(pop)
#endif
}
}  // namespace test
}  // namespace onnxruntime
