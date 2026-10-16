// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/allocation_planner_reuse_index.h"

namespace onnxruntime::allocation_planner_internal {

struct FreeBufferIndexTestPeer {
  static size_t Capacity(const FreeBufferIndex& index) { return index.slots_.size(); }
  static size_t Size(const FreeBufferIndex& index) { return index.size_; }
  static size_t Hash(size_t key) { return FreeBufferIndex::Hash(key); }
  static size_t NextCapacity(size_t capacity) { return FreeBufferIndex::NextCapacity(capacity); }
};

namespace {

TEST(FreeBufferIndexTest, EmptyZeroAndMaxWords) {
  FreeBufferIndex index;
  const auto max = std::numeric_limits<size_t>::max();
  EXPECT_EQ(index.Find(0), nullptr);
  EXPECT_EQ(index.Find(max), nullptr);
  EXPECT_FALSE(index.GetOrCreate(0).has_value());
  index.GetOrCreate(0) = max;
  EXPECT_FALSE(index.GetOrCreate(max).has_value());
  index.GetOrCreate(max) = 0;
  ASSERT_NE(index.Find(0), nullptr);
  ASSERT_NE(index.Find(max), nullptr);
  EXPECT_EQ(*index.Find(0), max);
  EXPECT_EQ(*index.Find(max), 0u);
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), 2u);
}

TEST(FreeBufferIndexTest, FullWordCollisionsKeepConsumedProbeChains) {
  // 32 words in eight initial buckets guarantee at least four colliding words.
  // Equal low halves also detect accidentally narrowed key comparisons.
  constexpr int shift = std::numeric_limits<size_t>::digits / 2;
  std::array<std::vector<size_t>, 8> groups;
  for (size_t i = 0; i < 32; ++i) {
    size_t key = (i << shift) | 7;
    groups[FreeBufferIndexTestPeer::Hash(key) & 7].push_back(key);
  }
  const auto group = std::find_if(groups.begin(), groups.end(),
                                  [](const auto& keys) { return keys.size() >= 4; });
  ASSERT_NE(group, groups.end());
  FreeBufferIndex index;
  for (size_t i = 0; i < 4; ++i) index.GetOrCreate((*group)[i]) = i;
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), 8u);
  index.Find((*group)[0])->reset();
  index.Find((*group)[1])->reset();
  for (size_t i = 2; i < 4; ++i) {
    ASSERT_NE(index.Find((*group)[i]), nullptr);
    EXPECT_EQ(*index.Find((*group)[i]), i);
  }
  EXPECT_FALSE(index.GetOrCreate((*group)[0]).has_value());
  index.GetOrCreate((*group)[0]) = std::numeric_limits<size_t>::max();
  EXPECT_EQ(*index.Find((*group)[0]), std::numeric_limits<size_t>::max());
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), 4u);
}

TEST(FreeBufferIndexTest, DuplicateAndReaddedHeadsDoNotGrowAtLoadLimit) {
  FreeBufferIndex index;
  for (size_t i = 0; i < 6; ++i) index.GetOrCreate(i) = i + 10;
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), 8u);
  EXPECT_EQ(index.GetOrCreate(2), 12u);
  index.Find(2)->reset();
  EXPECT_FALSE(index.GetOrCreate(2).has_value());
  index.GetOrCreate(2) = 99;
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), 8u);
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), 6u);
  index.Find(4)->reset();
  index.GetOrCreate(6) = 16;
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), 16u);
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), 7u);
  EXPECT_EQ(*index.Find(2), 99u);
  ASSERT_NE(index.Find(4), nullptr);
  EXPECT_FALSE(index.Find(4)->has_value());
}

TEST(FreeBufferIndexTest, WrappingProbeChainRetainsConsumedEntries) {
  std::vector<size_t> keys;
  for (size_t key = 0; key < 4096 && keys.size() < 4; ++key) {
    if ((FreeBufferIndexTestPeer::Hash(key) & 7) == 7) keys.push_back(key);
  }
  ASSERT_EQ(keys.size(), 4u);
  FreeBufferIndex index;
  for (size_t i = 0; i < 3; ++i) index.GetOrCreate(keys[i]) = i;
  index.Find(keys[0])->reset();
  EXPECT_EQ(*index.Find(keys[2]), 2u);
  EXPECT_EQ(index.Find(keys[3]), nullptr);
  index.GetOrCreate(keys[3]) = 3;
  EXPECT_EQ(*index.Find(keys[3]), 3u);
  EXPECT_FALSE(index.GetOrCreate(keys[0]).has_value());
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), 8u);
}

TEST(FreeBufferIndexTest, GrowthPreservesEveryWordAndConsumedBucket) {
  FreeBufferIndex index;
  std::map<size_t, std::optional<size_t>> expected;
  const auto max = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < 2048; ++i) {
    const size_t key = i == 0 ? max : i - 1;
    auto& head = index.GetOrCreate(key);
    if (i % 5 == 0) {
      head.reset();
      expected[key] = std::nullopt;
    } else {
      const size_t value = i == 1 ? max : i == 2 ? 0
                                                 : max - i;
      head = value;
      expected[key] = value;
    }
    const auto capacity = FreeBufferIndexTestPeer::Capacity(index);
    EXPECT_EQ(capacity & (capacity - 1), 0u);
    EXPECT_LE(FreeBufferIndexTestPeer::Size(index), capacity - capacity / 4);
  }
  for (const auto& [key, head] : expected) {
    ASSERT_NE(index.Find(key), nullptr);
    EXPECT_EQ(*index.Find(key), head);
  }
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), expected.size());
  EXPECT_EQ(index.Find(4096), nullptr);
}

TEST(FreeBufferIndexTest, ClearRemovesAllKeysAndStaleHeads) {
  FreeBufferIndex index;
  for (size_t i = 0; i < 30; ++i) index.GetOrCreate(i) = i;
  index.Find(3)->reset();
  const auto capacity = FreeBufferIndexTestPeer::Capacity(index);
  index.clear();
  EXPECT_EQ(FreeBufferIndexTestPeer::Size(index), 0u);
  EXPECT_EQ(FreeBufferIndexTestPeer::Capacity(index), capacity);
  for (size_t i = 0; i < 30; ++i) {
    EXPECT_EQ(index.Find(i), nullptr);
    EXPECT_FALSE(index.GetOrCreate(i).has_value());
    index.GetOrCreate(i) = i + 50;
  }
  for (size_t i = 0; i < 30; ++i) EXPECT_EQ(*index.Find(i), i + 50);
}

#if GTEST_HAS_EXCEPTIONS && !defined(ORT_NO_EXCEPTIONS)
TEST(FreeBufferIndexTest, CheckedCapacityGrowth) {
  EXPECT_EQ(FreeBufferIndexTestPeer::NextCapacity(0), 8u);
  EXPECT_EQ(FreeBufferIndexTestPeer::NextCapacity(8), 16u);
  EXPECT_THROW(FreeBufferIndexTestPeer::NextCapacity(std::numeric_limits<size_t>::max()),
               OnnxRuntimeException);
}
#endif

}  // namespace
}  // namespace onnxruntime::allocation_planner_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
