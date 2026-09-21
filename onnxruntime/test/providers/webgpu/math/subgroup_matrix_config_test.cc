// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <optional>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"

namespace onnxruntime {
namespace test {

TEST(SubgroupMatrixConfigTest, RequiredSubgroupSizeCompatibility) {
  using webgpu::IsSubgroupSizeSupported;

  EXPECT_TRUE(IsSubgroupSizeSupported(32, 32, 32, false));  // NVIDIA and Apple fixed-size adapters
  EXPECT_TRUE(IsSubgroupSizeSupported(32, 64, 32, true));   // AMD variable-size adapter
  EXPECT_TRUE(IsSubgroupSizeSupported(16, 32, 32, true));   // Intel variable-size adapter

  EXPECT_FALSE(IsSubgroupSizeSupported(32, 64, 32, false));
  EXPECT_FALSE(IsSubgroupSizeSupported(64, 64, 32, true));
  EXPECT_FALSE(IsSubgroupSizeSupported(16, 16, 32, true));
  EXPECT_FALSE(IsSubgroupSizeSupported(64, 32, 32, true));
}

TEST(SubgroupMatrixConfigTest, OperationPreferenceSelectsFromAllCandidates) {
  using webgpu::supported_subgroup_matrix_configs;
  using webgpu::detail::SelectSubgroupMatrixConfigFromCandidates;

  const auto find_index = [](uint32_t m, uint32_t n, uint32_t k, uint32_t subgroup_size) {
    for (size_t i = 0; i < supported_subgroup_matrix_configs.size(); ++i) {
      const auto& config = supported_subgroup_matrix_configs[i];
      if (config.Is(m, n, k) && config.subgroupSize == subgroup_size) {
        return static_cast<int32_t>(i);
      }
    }
    return int32_t{-1};
  };

  const int32_t matmul_nbits = find_index(16, 16, 16, 32);
  const int32_t intel = find_index(8, 16, 16, 32);
  const int32_t apple = find_index(8, 8, 8, 32);
  ASSERT_GE(matmul_nbits, 0);
  ASSERT_GE(intel, 0);
  ASSERT_GE(apple, 0);

  // Deliberately scramble candidate order. The operation preference, rather than candidate or
  // global table order, must decide which valid kernel wins.
  const InlinedVector<int32_t, 3> candidates{matmul_nbits, apple, intel};
  const auto prefer_intel =
      SelectSubgroupMatrixConfigFromCandidates(candidates, {{8, 16, 16, 32}, {16, 16, 16, 32}});
  ASSERT_TRUE(prefer_intel.has_value());
  EXPECT_EQ(*prefer_intel, intel);

  const auto prefer_matmul_nbits =
      SelectSubgroupMatrixConfigFromCandidates(candidates, {{16, 16, 16, 32}, {8, 16, 16, 32}});
  ASSERT_TRUE(prefer_matmul_nbits.has_value());
  EXPECT_EQ(*prefer_matmul_nbits, matmul_nbits);

  EXPECT_EQ(SelectSubgroupMatrixConfigFromCandidates(candidates, {{8, 8, 8, 64}}), std::nullopt);
}

}  // namespace test
}  // namespace onnxruntime
