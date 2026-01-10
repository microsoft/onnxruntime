// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_key_value_pairs.h"

#include <algorithm>
#include <iterator>

#include "gtest/gtest.h"

namespace onnxruntime::test {

namespace {

// Verify that the OrtKeyValuePairs internal containers are all consistent.
void CheckConsistency(const OrtKeyValuePairs& kvps) {
  ASSERT_EQ(kvps.Keys().size(), kvps.Entries().size());
  ASSERT_EQ(kvps.Values().size(), kvps.Entries().size());

  for (const auto& [k, v] : kvps.Entries()) {
    auto key_it = std::find(kvps.Keys().begin(), kvps.Keys().end(), k.c_str());
    ASSERT_NE(key_it, kvps.Keys().end());

    const auto entry_idx = std::distance(kvps.Keys().begin(), key_it);
    ASSERT_EQ(kvps.Values()[entry_idx], v.c_str());
  }
}

}  // namespace

TEST(OrtKeyValuePairsTest, BasicUsage) {
  const auto kvp_entry_map = std::map<std::string, std::string>{
      {"a", "1"}, {"b", "2"}, {"c", "3"}};

  OrtKeyValuePairs kvps{};
  kvps.CopyFromMap(kvp_entry_map);
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries(), kvp_entry_map);

  kvps.Add("d", "4");
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries().size(), 4);

  kvps.Remove("c");
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries().size(), 3);
}

TEST(OrtKeyValuePairsTest, CopyAndMove) {
  const auto kvp_entry_map = std::map<std::string, std::string>{
      {"a", "1"}, {"b", "2"}, {"c", "3"}};

  OrtKeyValuePairs kvps0{};
  kvps0.CopyFromMap(kvp_entry_map);
  CheckConsistency(kvps0);

  OrtKeyValuePairs kvps1 = kvps0;
  CheckConsistency(kvps1);
  ASSERT_EQ(kvps1.Entries(), kvps0.Entries());

  OrtKeyValuePairs kvps2 = std::move(kvps1);
  CheckConsistency(kvps1);
  CheckConsistency(kvps2);
  ASSERT_TRUE(kvps1.Entries().empty());
  ASSERT_EQ(kvps2.Entries(), kvps0.Entries());
}

TEST(OrtKeyValuePairsTest, Overwrite) {
  OrtKeyValuePairs kvps{};

  kvps.Add("a", "1");
  CheckConsistency(kvps);

  kvps.Add("a", "2");
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Values().size(), 1);
  ASSERT_STREQ(kvps.Values()[0], "2");
}

TEST(OrtKeyValuePairsTest, IgnoredInput) {
  OrtKeyValuePairs kvps{};

  kvps.Add(nullptr, "1");
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries().size(), size_t{0});

  kvps.Add("a", nullptr);
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries().size(), size_t{0});

  kvps.Add("", "1");  // empty key is ignored
  CheckConsistency(kvps);
  ASSERT_EQ(kvps.Entries().size(), size_t{0});
}

}  // namespace onnxruntime::test
