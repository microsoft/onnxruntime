// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/make_string.h"
#include "core/common/parse_string.h"

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

}  // namespace test
}  // namespace onnxruntime
