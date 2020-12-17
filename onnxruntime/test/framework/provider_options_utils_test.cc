// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/provider_options_utils.h"

#include "gtest/gtest.h"

#include "asserts.h"

namespace onnxruntime {
namespace test {

TEST(ProviderOptionsUtilsTest, ProviderOptionsParser) {
  int i;
  bool b;
  ProviderOptionsParser parser{};
  parser.AddAssignmentToReference("int", i);
  parser.AddAssignmentToReference("bool", b);

  // adding same option again should throw
  ASSERT_THROW(parser.AddAssignmentToReference("int", i), OnnxRuntimeException);

  ASSERT_STATUS_OK(parser.Parse({{"int", "3"}, {"bool", "true"}}));
  EXPECT_EQ(i, 3);
  EXPECT_EQ(b, true);

  ASSERT_FALSE(parser.Parse({{"unknown option", "some value"}}).IsOK());
}

}  // namespace test
}  // namespace onnxruntime
