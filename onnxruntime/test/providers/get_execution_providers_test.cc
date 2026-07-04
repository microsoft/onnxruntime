// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/get_execution_providers.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "gtest/gtest.h"

#include "core/graph/constants.h"

namespace onnxruntime {
namespace test {

TEST(GetExecutionProvidersTest, CpuEpAlwaysLast) {
  const auto check = [](const std::vector<std::string>& providers) {
    ASSERT_FALSE(providers.empty());
    EXPECT_EQ(providers.back(), kCpuExecutionProvider);
  };

  check(GetAllExecutionProviderNames());
  check(GetAvailableExecutionProviderNames());
}

TEST(GetExecutionProvidersTest, ConsistentOrdering) {
  const auto& all = GetAllExecutionProviderNames();
  const auto& available = GetAvailableExecutionProviderNames();
  std::vector<std::string> available_from_all{};
  std::copy_if(
      all.begin(), all.end(),
      std::back_inserter(available_from_all),
      [&available](const std::string& value) {
        return std::find(available.begin(), available.end(), value) != available.end();
      });

  EXPECT_EQ(available, available_from_all);
}

TEST(GetExecutionProvidersTest, NoDuplicates) {
  const auto check = [](const std::vector<std::string>& providers) {
    const std::unordered_set<std::string> providers_set(providers.begin(), providers.end());
    EXPECT_EQ(providers.size(), providers_set.size());
  };

  check(GetAllExecutionProviderNames());
  check(GetAvailableExecutionProviderNames());
}

TEST(GetExecutionProvidersTest, CpuAlwaysUsable) {
  // CPU provider is always statically linked and should always be usable
  EXPECT_TRUE(IsExecutionProviderUsable(kCpuExecutionProvider));
}

TEST(GetExecutionProvidersTest, UnknownProviderNotUsable) {
  EXPECT_FALSE(IsExecutionProviderUsable("NonExistentExecutionProvider"));
}

TEST(GetExecutionProvidersTest, UsableIsSubsetOfAvailable) {
  const auto& available = GetAvailableExecutionProviderNames();
  const auto usable = GetUsableExecutionProviderNames();

  // Every usable provider must be in the available list
  for (const auto& provider : usable) {
    EXPECT_NE(std::find(available.begin(), available.end(), provider), available.end())
        << "Usable provider " << provider << " is not in available providers list";
  }

  // Usable count must be <= available count
  EXPECT_LE(usable.size(), available.size());
}

TEST(GetExecutionProvidersTest, UsableProvidersMaintainPriorityOrder) {
  const auto& available = GetAvailableExecutionProviderNames();
  const auto usable = GetUsableExecutionProviderNames();

  // Verify usable providers maintain the same relative ordering as available
  size_t last_available_idx = 0;
  for (const auto& provider : usable) {
    auto it = std::find(available.begin() + last_available_idx, available.end(), provider);
    ASSERT_NE(it, available.end())
        << "Usable provider " << provider << " not found in available list";
    last_available_idx = static_cast<size_t>(std::distance(available.begin(), it)) + 1;
  }
}

TEST(GetExecutionProvidersTest, CpuAlwaysLastInUsable) {
  const auto usable = GetUsableExecutionProviderNames();
  ASSERT_FALSE(usable.empty());
  EXPECT_EQ(usable.back(), kCpuExecutionProvider);
}

}  // namespace test
}  // namespace onnxruntime
