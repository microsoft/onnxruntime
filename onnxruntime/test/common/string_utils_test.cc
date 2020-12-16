// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/string_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
void TestSuccessfulParse(const std::string& input, const T& expected_value) {
  T value;
  ASSERT_TRUE(TryParse(input, value));
  EXPECT_EQ(value, expected_value);
}

template <typename T>
void TestFailedParse(const std::string& input) {
  T value;
  EXPECT_FALSE(TryParse(input, value));
}
}  // namespace

TEST(StringUtilsTest, TryParse) {
  TestSuccessfulParse("-1", -1);
  TestSuccessfulParse("42", 42u);
  TestSuccessfulParse("2.5", 2.5f);
  TestSuccessfulParse("1", true);
  TestSuccessfulParse("0", false);

  // out of range
  TestFailedParse<int16_t>("32768");
  TestFailedParse<uint32_t>("-1");
  TestFailedParse<float>("1e100");
  TestFailedParse<bool>("2");
  // invalid representation
  TestFailedParse<int32_t>("1.2");
  TestFailedParse<int32_t>("one");
  // leading or trailing characters
  TestFailedParse<int32_t>(" 1");
  TestFailedParse<int32_t>("1 ");
}

TEST(StringUtilsTest, TryParseString) {
  // when parsing a string as a string, allow leading and trailing whitespace
  const std::string s = "  this is a string! ";
  TestSuccessfulParse(s, s);
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

TEST(StringUtilsTest, MakeStringAndTryParseCustomType) {
  S s;
  s.i = 42;
  const auto str = MakeString(s);
  S parsed_s;
  ASSERT_TRUE(TryParse(str, parsed_s));
  ASSERT_EQ(parsed_s, s);
}

}  // namespace test
}  // namespace onnxruntime
