// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/path.h"

#include "gtest/gtest.h"

#include "test/util/include/gtest_utils.h"

namespace onnxruntime {
namespace test {

TEST(PathTest, Parse) {
  auto check_parse = [](
      const PathString& path_string,
      const PathString& expected_root,
      const std::vector<PathString>& expected_components) {
    Path p{};
    ASSERT_STATUS_OK(Path::Parse(path_string, p));
    EXPECT_EQ(p.GetComponents(), expected_components);
    EXPECT_EQ(p.GetRootPathString(), expected_root);
  };

  check_parse(
      ToPathString("i/am/relative"),
      ToPathString(""),
      {ToPathString("i"), ToPathString("am"), ToPathString("relative")});
#ifdef _WIN32
  check_parse(
      ToPathString("/i/am/rooted"),
      ToPathString("\\"),
      {ToPathString("i"), ToPathString("am"), ToPathString("rooted")});
  check_parse(
      ToPathString(R"(\\server\share\i\am\rooted)"),
      ToPathString(R"(\\server\share\)"),
      {ToPathString("i"), ToPathString("am"), ToPathString("rooted")});
  check_parse(
      ToPathString(R"(C:\i\am\rooted)"),
      ToPathString(R"(C:\)"),
      {ToPathString("i"), ToPathString("am"), ToPathString("rooted")});
  check_parse(
      ToPathString(R"(C:i\am\relative)"),
      ToPathString("C:"),
      {ToPathString("i"), ToPathString("am"), ToPathString("relative")});
#else // POSIX
  check_parse(
      ToPathString("/i/am/rooted"),
      ToPathString("/"),
      {ToPathString("i"), ToPathString("am"), ToPathString("rooted")});
  check_parse(
      ToPathString("//root_name/i/am/rooted"),
      ToPathString("//root_name/"),
      {ToPathString("i"), ToPathString("am"), ToPathString("rooted")});
#endif
}

TEST(PathTest, Normalization) {
  auto check_normalized = [](
      const PathString& path_string,
      const PathString& expected_normalized_path_string) {
    Path p{};
    ASSERT_STATUS_OK(Path::Parse(path_string, p));
    Path p_expected_normalized{};
    ASSERT_STATUS_OK(Path::Parse(expected_normalized_path_string, p_expected_normalized));
    EXPECT_EQ(p.Normalized().ToPathString(), p_expected_normalized.ToPathString());
  };

  check_normalized(ToPathString("/a/b/./c/../../d/../e"), ToPathString("/a/e"));
  check_normalized(ToPathString("a/b/./c/../../d/../e"), ToPathString("a/e"));
  check_normalized(ToPathString("/../a/../../b"), ToPathString("/b"));
  check_normalized(ToPathString("../a/../../b"), ToPathString("../../b"));
}

TEST(PathTest, RelativePath) {
  auto check_relative = [](
      const PathString& src,
      const PathString& dst,
      const PathString& expected_rel) {
    Path p_src, p_dst, p_expected_rel, p_rel;
    ASSERT_STATUS_OK(Path::Parse(src, p_src));
    ASSERT_STATUS_OK(Path::Parse(dst, p_dst));
    ASSERT_STATUS_OK(Path::Parse(expected_rel, p_expected_rel));

    ASSERT_STATUS_OK(RelativePath(p_src, p_dst, p_rel));
    EXPECT_EQ(p_rel.ToPathString(), p_expected_rel.ToPathString());
  };

  check_relative(
      ToPathString("/a/b/c/d/e"), ToPathString("/a/b/c/d/e/f/g/h"),
      ToPathString("f/g/h"));
  check_relative(
      ToPathString("/a/b/c/d/e"), ToPathString("/a/b/f/g/h/i"),
      ToPathString("../../../f/g/h/i"));
  check_relative(
      ToPathString("a/b/../c/../d"), ToPathString("e/./f/../g/h"),
      ToPathString("../../e/g/h"));
}

TEST(PathTest, RelativePathFailure) {
  auto check_relative_failure = [](
      const PathString& src,
      const PathString& dst) {
    Path p_src, p_dst, p_rel;
    ASSERT_STATUS_OK(Path::Parse(src, p_src));
    ASSERT_STATUS_OK(Path::Parse(dst, p_dst));

    EXPECT_FALSE(RelativePath(p_src, p_dst, p_rel).IsOK());
  };

  check_relative_failure(ToPathString("/rooted"), ToPathString("relative"));
  check_relative_failure(ToPathString("relative"), ToPathString("/rooted"));
#ifdef _WIN32
  check_relative_failure(ToPathString("C:/a"), ToPathString("D:/a"));
#else // POSIX
  check_relative_failure(ToPathString("//root_0/a"), ToPathString("//root_1/a"));
#endif
}

}  // namespace test
}  // namespace onnxruntime
