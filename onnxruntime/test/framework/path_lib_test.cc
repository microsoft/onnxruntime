// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "gtest/gtest.h"

#include "core/platform/env.h"
#include "test/framework/test_utils.h"
#include "core/framework/path_lib.h"  // TODO fix include order dependency, path_lib.h should be first
#include "test/util/include/gtest_utils.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime {
namespace test {

#define PATH_EXPECT(X, Y)                                            \
  {                                                                  \
    auto a = ORT_TSTR(X);                                            \
    std::basic_string<ORTCHAR_T> ret;                                \
    auto st = onnxruntime::GetDirNameFromFilePath(ORT_TSTR(Y), ret); \
    ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();                     \
    ASSERT_EQ(a, ret);                                               \
  }

#ifdef _WIN32
TEST(PathTest, simple) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\a.txt");
}

TEST(PathTest, trailing_slash) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\system32\\");
}

TEST(PathTest, trailing_slash2) {
  PATH_EXPECT("C:\\Windows", "C:\\Windows\\\\system32\\");
}

TEST(PathTest, windows_root) {
  PATH_EXPECT("C:\\", "C:\\");
}
TEST(PathTest, root) {
  PATH_EXPECT("\\", "\\");
}
#else
TEST(PathTest, simple) {
  PATH_EXPECT("/Windows", "/Windows/a.txt");
}

TEST(PathTest, trailing_slash) {
  PATH_EXPECT("/Windows", "/Windows/system32/");
}

TEST(PathTest, trailing_slash2) {
  PATH_EXPECT("/Windows", "/Windows//system32/");
}

TEST(PathTest, root) {
  PATH_EXPECT("/", "/");
}
#endif
TEST(PathTest, single) {
  PATH_EXPECT(".", "abc");
}

TEST(PathTest, dot) {
  PATH_EXPECT(".", ".");
}

TEST(PathTest, dot_dot) {
  PATH_EXPECT(".", "..");
}

#undef PATH_EXPECT

namespace {

void CheckGetRelativePath(
    const PathString& src, const PathString& dst, const PathString& expected) {
  PathString result;
  ASSERT_STATUS_OK(GetRelativePath(NormalizePathSeparators(src), NormalizePathSeparators(dst), result));
  ASSERT_EQ(result, NormalizePathSeparators(expected));
}

}  // namespace

TEST(PathTest, GetRelativePathBasic) {
  TemporaryDirectory tmp_dir{ORT_TSTR("path_test")};
  auto get_tmp_dir_path = [&tmp_dir](const PathString& relative_path) {
    return ConcatPathComponent<PathChar>(tmp_dir.Path(), NormalizePathSeparators(relative_path));
  };
  ASSERT_STATUS_OK(Env::Default().CreateFolder(get_tmp_dir_path(ORT_TSTR("a/b/c"))));
  ASSERT_STATUS_OK(Env::Default().CreateFolder(get_tmp_dir_path(ORT_TSTR("a/d/e/f"))));
  CheckGetRelativePath(
      get_tmp_dir_path(ORT_TSTR("a/b/c")), get_tmp_dir_path(ORT_TSTR("a/d/e/f")),
      ORT_TSTR("../../d/e/f"));
  CheckGetRelativePath(
      get_tmp_dir_path(ORT_TSTR("a/d/e/f")), get_tmp_dir_path(ORT_TSTR("a/b")),
      ORT_TSTR("../../../b"));
}
}  // namespace test
}  // namespace onnxruntime
