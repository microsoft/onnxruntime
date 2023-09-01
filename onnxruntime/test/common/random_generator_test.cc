// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include <vector>

namespace onnxruntime {
namespace test {

TEST(TensorGenerator, DiscreteFloat) {
  FixedPatternValueGenerator random{};
  const std::vector<int64_t> shape = {2, 3};
  std::vector<float> data = random.Discrete<float>(shape, AsSpan({-1.f, 0.f, 1.f}));

  ASSERT_EQ(data.size(), static_cast<size_t>(6));
  for (float value : data) {
    EXPECT_TRUE(value == -1.f || value == 0.f || value == 1.f);
  }
}

TEST(TensorGenerator, DiscreteInt) {
  FixedPatternValueGenerator random{};
  const std::vector<int64_t> shape = {2, 3};
  std::vector<int> data = random.Discrete<int>(shape, AsSpan({-1, 0, 1}));

  ASSERT_EQ(data.size(), static_cast<size_t>(6));
  for (int value : data) {
    EXPECT_TRUE(value == -1 || value == 0 || value == 1);
  }
}

// Tests for Circular
TEST(TensorGenerator, CircularFloat) {
  FixedPatternValueGenerator random{};
  const std::vector<int64_t> shape = {3, 2};
  std::vector<float> data = random.Circular<float>(shape, AsSpan({-1.f, 0.f, 1.f}));

  ASSERT_EQ(data.size(), static_cast<size_t>(6));
  EXPECT_EQ(data[0], -1.f);
  EXPECT_EQ(data[1], 0.f);
  EXPECT_EQ(data[2], 1.f);
  EXPECT_EQ(data[3], -1.f);
  EXPECT_EQ(data[4], 0.f);
  EXPECT_EQ(data[5], 1.f);
}

TEST(TensorGenerator, CircularInt) {
  FixedPatternValueGenerator random{};
  const std::vector<int64_t> shape = {3, 2};
  std::vector<int> data = random.Circular<int>(shape, AsSpan({-1, 0, 1}));

  ASSERT_EQ(data.size(), static_cast<size_t>(6));
  EXPECT_EQ(data[0], -1);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 1);
  EXPECT_EQ(data[3], -1);
  EXPECT_EQ(data[4], 0);
  EXPECT_EQ(data[5], 1);
}

TEST(TensorGenerator, CircularBool) {
  FixedPatternValueGenerator random{};
  const std::vector<int64_t> shape = {3, 2};
  std::vector<bool> data = random.Circular<bool>(shape, AsSpan({false, true}));

  ASSERT_EQ(data.size(), static_cast<size_t>(6));
  EXPECT_EQ(data[0], false);
  EXPECT_EQ(data[1], true);
  EXPECT_EQ(data[2], false);
  EXPECT_EQ(data[3], true);
  EXPECT_EQ(data[4], false);
  EXPECT_EQ(data[5], true);
}

}  // namespace test
}  // namespace onnxruntime
