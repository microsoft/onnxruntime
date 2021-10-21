// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/provider_options_utils.h"

#include "gtest/gtest.h"

#include "asserts.h"

namespace onnxruntime {
namespace test {

namespace {
enum class TestEnum {
  A,
  Unmapped,
};

const EnumNameMapping<TestEnum> test_enum_mapping{
    {TestEnum::A, "A"},
};
}  // namespace

TEST(ProviderOptionsUtilsTest, ProviderOptionsParser) {
  int i;
  bool b;
  TestEnum e;
  ProviderOptionsParser parser{};
  parser.AddAssignmentToReference("int", i);
  parser.AddAssignmentToReference("bool", b);
  parser.AddAssignmentToEnumReference("enum", test_enum_mapping, e);

  // adding same option again should throw
  ASSERT_THROW(parser.AddAssignmentToReference("int", i), OnnxRuntimeException);

  ASSERT_STATUS_OK(parser.Parse({{"int", "3"}, {"bool", "true"}, {"enum", "A"}}));
  EXPECT_EQ(i, 3);
  EXPECT_EQ(b, true);
  EXPECT_EQ(e, TestEnum::A);

  ASSERT_FALSE(parser.Parse({{"unknown option", "some value"}}).IsOK());
}

TEST(ProviderOptionsUtilsTest, EnumToName) {
  std::string name;
  ASSERT_STATUS_OK(EnumToName(test_enum_mapping, TestEnum::A, name));
  EXPECT_EQ(name, "A");
  ASSERT_FALSE(EnumToName(test_enum_mapping, TestEnum::Unmapped, name).IsOK());
}

TEST(ProviderOptionsUtilsTest, NameToEnum) {
  TestEnum value;
  ASSERT_STATUS_OK(NameToEnum(test_enum_mapping, "A", value));
  EXPECT_EQ(value, TestEnum::A);
  ASSERT_FALSE(NameToEnum(test_enum_mapping, "invalid", value).IsOK());
}

}  // namespace test
}  // namespace onnxruntime
