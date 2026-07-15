// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>

#include "gtest/gtest.h"

#include "core/common/path_utils.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime {
namespace test {

TEST(PathUtilsTest, GetDirOrParentPath) {
  TemporaryDirectory temp_dir{ORT_TSTR("path_utils_test")};
  const std::filesystem::path directory{temp_dir.Path()};

  EXPECT_TRUE(path_utils::GetDirOrParentPath({}).empty());
  EXPECT_EQ(path_utils::GetDirOrParentPath(directory), directory);
  EXPECT_EQ(path_utils::GetDirOrParentPath(directory / "model.onnx"), directory);
}

TEST(PathUtilsTest, GetPathWithStemSuffix) {
  TemporaryDirectory temp_dir{ORT_TSTR("path_utils_test")};
  const std::filesystem::path directory{temp_dir.Path()};
  const std::string directory_string = PathToUTF8String(directory.native());
  const std::filesystem::path explicit_path = directory / "custom_context.onnx";
  const std::string explicit_path_string = PathToUTF8String(explicit_path.native());

  EXPECT_EQ(path_utils::GetPathWithStemSuffix("", "model.onnx", "_ctx.onnx"), "model_ctx.onnx");
  EXPECT_EQ(path_utils::GetPathWithStemSuffix("", "model.v1.onnx", "_ctx.onnx"), "model.v1_ctx.onnx");
  EXPECT_EQ(path_utils::GetPathWithStemSuffix(explicit_path_string, "model.onnx", "_ctx.onnx"),
            explicit_path_string);
  EXPECT_EQ(path_utils::GetPathWithStemSuffix(directory_string, "model.onnx", "_ctx.onnx"),
            (directory / "model_ctx.onnx").string());
}

TEST(PathUtilsTest, IsAbsolutePath) {
  const std::filesystem::path absolute_path = std::filesystem::absolute("cache");

  EXPECT_TRUE(path_utils::IsAbsolutePath(PathToUTF8String(absolute_path.native())));
  EXPECT_FALSE(path_utils::IsAbsolutePath("cache/engine"));
  EXPECT_FALSE(path_utils::IsAbsolutePath(""));
}

TEST(PathUtilsTest, IsRelativePathToParentPath) {
  EXPECT_TRUE(path_utils::IsRelativePathToParentPath("../cache"));
#ifdef _WIN32
  EXPECT_FALSE(path_utils::IsRelativePathToParentPath("cache/../engine"));
#else
  EXPECT_TRUE(path_utils::IsRelativePathToParentPath("cache/../engine"));
#endif
  EXPECT_TRUE(path_utils::IsRelativePathToParentPath("cache..engine"));
  EXPECT_FALSE(path_utils::IsRelativePathToParentPath("cache/engine"));
  EXPECT_FALSE(path_utils::IsRelativePathToParentPath(""));
}

}  // namespace test
}  // namespace onnxruntime
