// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
void TestSuccessfulParse(const std::string& input, const T& expected_value) {
  T value;
  ASSERT_TRUE(TryParseStringWithClassicLocale(input, value));
  EXPECT_EQ(value, expected_value);
}

template <typename T>
void TestFailedParse(const std::string& input) {
  T value;
  EXPECT_FALSE(TryParseStringWithClassicLocale(input, value));
}
}  // namespace

TEST(StringUtilsTest, TryParseStringWithClassicLocale) {
  TestSuccessfulParse("-1", -1);
  TestSuccessfulParse("42", 42u);
  TestSuccessfulParse("2.5", 2.5f);

  // out of range
  TestFailedParse<int16_t>("32768");
  TestFailedParse<uint32_t>("-1");
  TestFailedParse<float>("1e100");
  // invalid representation
  TestFailedParse<int32_t>("1.2");
  TestFailedParse<int32_t>("one");
  // leading or trailing characters
  TestFailedParse<int32_t>(" 1");
  TestFailedParse<int32_t>("1 ");
}

TEST(StringUtilsTest, TryParseStringAsString) {
  // when parsing a string as a string, allow leading and trailing whitespace
  const std::string s = "  this is a string! ";
  TestSuccessfulParse(s, s);
}

TEST(StringUtilsTest, TryParseStringAsBool) {
  TestSuccessfulParse("True", true);
  TestSuccessfulParse("1", true);
  TestSuccessfulParse("False", false);
  TestSuccessfulParse("0", false);

  TestFailedParse<bool>("2");
}

namespace {
struct S {
  int i{};

  bool operator==(const S& other) const {
    return i == other.i;
  }
};

std::ostream& operator<<(std::ostream& os, const S& s) {
  os << "S " << s.i;
  return os;
}

std::istream& operator>>(std::istream& is, S& s) {
  std::string struct_name;
  is >> struct_name >> s.i;
  if (struct_name != "S") {
    is.setstate(std::ios_base::failbit);
  }
  return is;
}
}  // namespace

TEST(StringUtilsTest, MakeStringAndTryParseStringWithCustomType) {
  S s;
  s.i = 42;
  const auto str = MakeStringWithClassicLocale(s);
  S parsed_s;
  ASSERT_TRUE(TryParseStringWithClassicLocale(str, parsed_s));
  ASSERT_EQ(parsed_s, s);
}

TEST(StringUtilsTest, SplitString) {
  auto run_test = [](const std::string& string_to_split, const std::string& delimiter,
                     const std::vector<std::string>& expected_substrings_with_empty) {
    SCOPED_TRACE(MakeString("string_to_split: \"", string_to_split, "\", delimiter: \"", delimiter, "\""));

    auto test_split = [&](const std::vector<std::string>& expected_substrings, bool keep_empty) {
      SCOPED_TRACE(MakeString("keep_empty: ", keep_empty));

      const auto actual_substrings = utils::SplitString(string_to_split, delimiter, keep_empty);
      ASSERT_EQ(actual_substrings.size(), expected_substrings.size());
      for (size_t i = 0; i < actual_substrings.size(); ++i) {
        EXPECT_EQ(actual_substrings[i], expected_substrings[i]) << "i=" << i;
      }
    };

    test_split(expected_substrings_with_empty, true);

    const std::vector<std::string> expected_substrings_without_empty = [&]() {
      std::vector<std::string> result = expected_substrings_with_empty;
      result.erase(std::remove_if(result.begin(), result.end(),
                                  [](const std::string& value) { return value.empty(); }),
                   result.end());
      return result;
    }();
    test_split(expected_substrings_without_empty, false);
  };

  run_test("a,b,c", ",", {"a", "b", "c"});
  run_test(",a,,b,,,c,", ",", {"", "a", "", "b", "", "", "c", ""});
  run_test("one_delimiter_two_delimiter_", "_delimiter_", {"one", "two", ""});
  run_test("aaaaaaa", "aa", {"", "", "", "a"});
  run_test("abcabaabc", "abc", {"", "aba", ""});
  run_test("leading,", ",", {"leading", ""});
  run_test(",trailing", ",", {"", "trailing"});
  run_test("", ",", {""});
  run_test(",", ",", {"", ""});
}

#ifndef ORT_NO_EXCEPTIONS
TEST(StringUtilsTest, SplitStringWithEmptyDelimiter) {
  EXPECT_THROW(utils::SplitString("a", ""), OnnxRuntimeException);
}
#endif

}  // namespace test
}  // namespace onnxruntime
