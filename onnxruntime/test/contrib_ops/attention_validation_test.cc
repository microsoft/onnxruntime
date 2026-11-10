// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/bert/attention_validation.h"

namespace onnxruntime {
namespace test {

TEST(ContribOpAttentionValidationTest, WeightDimensionBounds) {
  using contrib::attention::CheckAttentionWeights;
  constexpr int64_t max = (std::numeric_limits<int>::max)();
  std::array<int64_t, 3> hidden_sizes{};
  const std::array<int64_t, 2> valid_dims{4, 12};
  ASSERT_TRUE(CheckAttentionWeights(valid_dims, {}, 2, false, hidden_sizes).IsOK());
  EXPECT_EQ(hidden_sizes, (std::array<int64_t, 3>{4, 4, 4}));
  for (const auto& dims : {std::array<int64_t, 2>{-1, 12}, {max + 1, 12}, {4, max + 1}}) {
    EXPECT_FALSE(CheckAttentionWeights(dims, {}, 2, false, hidden_sizes).IsOK());
    EXPECT_EQ(hidden_sizes, (std::array<int64_t, 3>{4, 4, 4}));
  }
  EXPECT_FALSE(CheckAttentionWeights(valid_dims, {}, 0, false, hidden_sizes).IsOK());
  const auto invalid_sizes = {
      std::vector<int64_t>{4},
      {4, 4},
      {4, 4, 4, 4},
      {-2, -2, 16},
      {0, 0, 12},
      {max + 1, max + 1, 4},
  };
  for (const auto& sizes : invalid_sizes) {
    SCOPED_TRACE(testing::PrintToString(sizes));
    EXPECT_FALSE(CheckAttentionWeights(valid_dims, sizes, 2, false, hidden_sizes).IsOK());
    EXPECT_EQ(hidden_sizes, (std::array<int64_t, 3>{4, 4, 4}));
  }
  const std::array<int64_t, 3> unequal_sizes{2, 2, 8};
  EXPECT_FALSE(CheckAttentionWeights(valid_dims, unequal_sizes, 2, true, hidden_sizes).IsOK());
  ASSERT_TRUE(CheckAttentionWeights(valid_dims, unequal_sizes, 2, false, hidden_sizes).IsOK());
  EXPECT_EQ(hidden_sizes, unequal_sizes);
}

TEST(ContribOpAttentionValidationTest, ProjectionDimensionBounds) {
  using contrib::attention::CheckAttentionProjectionSize;
  constexpr int64_t max = (std::numeric_limits<int>::max)();
  const int64_t max_elements = std::min<int64_t>(max, (std::numeric_limits<size_t>::max)() / sizeof(float));
  EXPECT_TRUE(CheckAttentionProjectionSize(1, 1, max_elements).IsOK());
  EXPECT_TRUE(CheckAttentionProjectionSize(0, max, max).IsOK());
  EXPECT_FALSE(CheckAttentionProjectionSize(1, 2, max_elements / 2 + 1).IsOK());
  EXPECT_FALSE(CheckAttentionProjectionSize(max, max, max).IsOK());
  EXPECT_FALSE(CheckAttentionProjectionSize(1, 1, max + 1).IsOK());
  EXPECT_FALSE(CheckAttentionProjectionSize(-1, 1, 12).IsOK());
}

TEST(ContribOpAttentionValidationTest, SequenceLengthBounds) {
  using contrib::attention::CheckAttentionSequenceLengths;
  constexpr int64_t max = (std::numeric_limits<int>::max)();
  int64_t total = -1;
  ASSERT_TRUE(CheckAttentionSequenceLengths(2, 1, 3, total).IsOK());
  EXPECT_EQ(total, 3);
  ASSERT_TRUE(CheckAttentionSequenceLengths(0, 3, 3, total).IsOK());
  EXPECT_EQ(total, 3);
  ASSERT_TRUE(CheckAttentionSequenceLengths(0, 0, 0, total).IsOK());
  EXPECT_EQ(total, 0);
  ASSERT_TRUE(CheckAttentionSequenceLengths(max, 0, std::nullopt, total).IsOK());
  EXPECT_EQ(total, max);
  EXPECT_FALSE(CheckAttentionSequenceLengths(2, -1, 4, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(0, 5, 4, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(2, 3, 4, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(0, 0, -1, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(max, 1, std::nullopt, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(0, max + 1, std::nullopt, total).IsOK());
  EXPECT_FALSE(CheckAttentionSequenceLengths(0, 0, max + 1, total).IsOK());
  EXPECT_EQ(total, max);
}

}  // namespace test
}  // namespace onnxruntime
