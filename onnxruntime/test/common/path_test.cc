// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/path.h"

#include "gtest/gtest.h"

#include "core/common/optional.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(PathTest, Parse) {
  auto check_parse =
      [](const std::string& path_string,
         const std::string& expected_root,
         const std::vector<std::string>& expected_components) {
        Path p{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(path_string), p));

        std::vector<PathString> expected_components_ps{};
        std::transform(
            expected_components.begin(), expected_components.end(),
            std::back_inserter(expected_components_ps),
            [](const std::string& s) { return ToPathString(s); });
        EXPECT_EQ(p.GetComponents(), expected_components_ps);
        EXPECT_EQ(p.GetRootPathString(), ToPathString(expected_root));
      };

  check_parse(
      "i/am/relative",
      "", {"i", "am", "relative"});
#ifdef _WIN32
  check_parse(
      "/i/am/rooted",
      R"(\)", {"i", "am", "rooted"});
  check_parse(
      R"(\\server\share\i\am\rooted)",
      R"(\\server\share\)", {"i", "am", "rooted"});
  check_parse(
      R"(C:\i\am\rooted)",
      R"(C:\)", {"i", "am", "rooted"});
  check_parse(
      R"(C:i\am\relative)",
      "C:", {"i", "am", "relative"});
#else  // POSIX
  check_parse(
      "/i/am/rooted",
      "/", {"i", "am", "rooted"});
  check_parse(
      "//root_name/i/am/rooted",
      "//root_name/", {"i", "am", "rooted"});
#endif
}

TEST(PathTest, ParseFailure) {
    auto check_parse_failure =
        [](const std::string& path_string) {
        Path p{};
        EXPECT_FALSE(Path::Parse(ToPathString(path_string), p).IsOK());
      };

#ifdef _WIN32
    check_parse_failure(R"(\\server_name_no_separator)");
    check_parse_failure(R"(\\server_name_no_share_name\)");
    check_parse_failure(R"(\\server_name\share_name_no_root_dir)");
#else  // POSIX
    check_parse_failure("//root_name_no_root_dir");
#endif
}

TEST(PathTest, IsEmpty) {
    auto check_empty =
        [](const std::string& path_string, bool is_empty) {
        Path p{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(path_string), p));

        EXPECT_EQ(p.IsEmpty(), is_empty);
      };

    check_empty("", true);
    check_empty(".", false);
    check_empty("/", false);
}

TEST(PathTest, IsAbsoluteOrRelative) {
  auto check_abs_or_rel =
      [](const std::string& path_string, bool is_absolute) {
        Path p{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(path_string), p));

        EXPECT_EQ(p.IsAbsolute(), is_absolute);
        EXPECT_EQ(p.IsRelative(), !is_absolute);
      };

  check_abs_or_rel("relative", false);
  check_abs_or_rel("", false);
#ifdef _WIN32
  check_abs_or_rel(R"(\root_relative)", false);
  check_abs_or_rel(R"(\)", false);
  check_abs_or_rel("C:drive_relative", false);
  check_abs_or_rel("C:", false);
  check_abs_or_rel(R"(C:\absolute)", true);
  check_abs_or_rel(R"(C:\)", true);
#else  // POSIX
  check_abs_or_rel("/absolute", true);
  check_abs_or_rel("/", true);
#endif
}

TEST(PathTest, ParentPath) {
  auto check_parent =
      [](const std::string path_string, const std::string& expected_parent_path_string) {
        Path p{}, p_expected_parent{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(path_string), p));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(expected_parent_path_string), p_expected_parent));

        EXPECT_EQ(p.ParentPath().ToPathString(), p_expected_parent.ToPathString());
      };

  check_parent("a/b", "a");
  check_parent("/a/b", "/a");
  check_parent("", "");
  check_parent("/", "/");
}

TEST(PathTest, Normalize) {
  auto check_normalize =
      [](const std::string& path_string,
         const std::string& expected_normalized_path_string) {
        Path p{}, p_expected_normalized{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(path_string), p));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(expected_normalized_path_string), p_expected_normalized));

        EXPECT_EQ(p.Normalize().ToPathString(), p_expected_normalized.ToPathString());
      };

  check_normalize("/a/b/./c/../../d/../e", "/a/e");
  check_normalize("a/b/./c/../../d/../e", "a/e");
  check_normalize("/../a/../../b", "/b");
  check_normalize("../a/../../b", "../../b");
  check_normalize("/a/..", "/");
  check_normalize("a/..", ".");
  check_normalize("", "");
  check_normalize("/", "/");
  check_normalize(".", ".");
}

TEST(PathTest, Append) {
  auto check_append =
      [](const std::string& a, const std::string& b, const std::string& expected_ab) {
        Path p_a{}, p_b{}, p_expected_ab{};
        ASSERT_STATUS_OK(Path::Parse(ToPathString(a), p_a));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(b), p_b));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(expected_ab), p_expected_ab));

        EXPECT_EQ(p_a.Append(p_b).ToPathString(), p_expected_ab.ToPathString());
      };

  check_append("/a/b", "c/d", "/a/b/c/d");
  check_append("/a/b", "/c/d", "/c/d");
  check_append("a/b", "c/d", "a/b/c/d");
  check_append("a/b", "/c/d", "/c/d");
#ifdef _WIN32
  check_append(R"(C:\a\b)", R"(c\d)", R"(C:\a\b\c\d)");
  check_append(R"(C:\a\b)", R"(\c\d)", R"(C:\c\d)");
  check_append(R"(C:\a\b)", R"(D:c\d)", R"(D:c\d)");
  check_append(R"(C:\a\b)", R"(D:\c\d)", R"(D:\c\d)");
  check_append(R"(C:a\b)", R"(c\d)", R"(C:a\b\c\d)");
  check_append(R"(C:a\b)", R"(\c\d)", R"(C:\c\d)");
  check_append(R"(C:a\b)", R"(D:c\d)", R"(D:c\d)");
  check_append(R"(C:a\b)", R"(D:\c\d)", R"(D:\c\d)");
#else  // POSIX
  check_append("//root_0/a/b", "c/d", "//root_0/a/b/c/d");
  check_append("//root_0/a/b", "/c/d", "/c/d");
  check_append("//root_0/a/b", "//root_1/c/d", "//root_1/c/d");
#endif
}

TEST(PathTest, RelativePath) {
  auto check_relative =
      [](const std::string& src,
         const std::string& dst,
         const std::string& expected_rel) {
        Path p_src, p_dst, p_expected_rel, p_rel;
        ASSERT_STATUS_OK(Path::Parse(ToPathString(src), p_src));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(dst), p_dst));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(expected_rel), p_expected_rel));

        ASSERT_STATUS_OK(RelativePath(p_src, p_dst, p_rel));
        EXPECT_EQ(p_rel.ToPathString(), p_expected_rel.ToPathString());
      };

  check_relative(
      "/a/b/c/d/e", "/a/b/c/d/e/f/g/h",
      "f/g/h");
  check_relative(
      "/a/b/c/d/e", "/a/b/f/g/h/i",
      "../../../f/g/h/i");
  check_relative(
      "a/b/../c/../d", "e/./f/../g/h",
      "../../e/g/h");
}

TEST(PathTest, RelativePathFailure) {
  auto check_relative_failure =
      [](const std::string& src,
         const std::string& dst) {
        Path p_src, p_dst, p_rel;
        ASSERT_STATUS_OK(Path::Parse(ToPathString(src), p_src));
        ASSERT_STATUS_OK(Path::Parse(ToPathString(dst), p_dst));

        EXPECT_FALSE(RelativePath(p_src, p_dst, p_rel).IsOK());
      };

  check_relative_failure("/rooted", "relative");
  check_relative_failure("relative", "/rooted");
#ifdef _WIN32
  check_relative_failure("C:/a", "D:/a");
#else  // POSIX
  check_relative_failure("//root_0/a", "//root_1/a");
#endif
}

TEST(PathTest, Concat) {
  auto check_concat =
      [](const optional<std::string>& a, const std::string& b, const std::string& expected_a, bool expect_throw = false) {
        Path p_a{}, p_expected_a{};
        if (a.has_value()) {
          ASSERT_STATUS_OK(Path::Parse(ToPathString(a.value()), p_a));
        }
        ASSERT_STATUS_OK(Path::Parse(ToPathString(expected_a), p_expected_a));

        if (expect_throw) {
          EXPECT_THROW(p_a.Concat(ToPathString(b)).ToPathString(), OnnxRuntimeException);
        } else {
          EXPECT_EQ(p_a.Concat(ToPathString(b)).ToPathString(), p_expected_a.ToPathString());
        }
      };

  check_concat({"/a/b"}, "c", "/a/bc");
  check_concat({"a/b"}, "cd", "a/bcd");
  check_concat({""}, "cd", "cd");
  check_concat({}, "c", "c");
#ifdef _WIN32
  check_concat({"a/b"}, R"(c\d)", "", true /* expect_throw */);
#else
  check_concat({"a/b"}, "c/d", "", true /* expect_throw */);
#endif
}

}  // namespace test
}  // namespace onnxruntime
