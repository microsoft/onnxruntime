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

}  // namespace test
}  // namespace onnxruntime
