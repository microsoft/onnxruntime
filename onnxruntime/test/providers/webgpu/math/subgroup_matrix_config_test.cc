// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <optional>

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/subgroup_matrix_config.h"

namespace onnxruntime {
namespace test {

TEST(SubgroupMatrixConfigTest, RequiredSubgroupSizeCompatibility) {
  using webgpu::IsSubgroupSizeSupported;

  EXPECT_TRUE(IsSubgroupSizeSupported(32, 32, 32, false));  // NVIDIA and Apple fixed-size adapters
  EXPECT_TRUE(IsSubgroupSizeSupported(64, 64, 64, false));  // Fixed wave64 adapters
  EXPECT_TRUE(IsSubgroupSizeSupported(32, 64, 32, true));   // AMD variable-size adapter
  EXPECT_TRUE(IsSubgroupSizeSupported(32, 64, 64, true));
  EXPECT_TRUE(IsSubgroupSizeSupported(16, 32, 32, true));  // Intel variable-size adapter

  EXPECT_FALSE(IsSubgroupSizeSupported(32, 64, 32, false));
  EXPECT_FALSE(IsSubgroupSizeSupported(32, 64, 64, false));
  EXPECT_FALSE(IsSubgroupSizeSupported(64, 64, 32, true));
  EXPECT_FALSE(IsSubgroupSizeSupported(16, 16, 32, true));
  EXPECT_FALSE(IsSubgroupSizeSupported(64, 32, 32, true));
}

TEST(SubgroupMatrixConfigTest, OperationPreferenceSelectsFromAdapterConfigs) {
  using webgpu::SubgroupMatrixConfig;
  using webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs;

  constexpr auto kF16 = wgpu::SubgroupMatrixComponentType::F16;
  constexpr auto kF32 = wgpu::SubgroupMatrixComponentType::F32;
  // Deliberately scramble adapter order. The operation preference must decide which valid kernel wins.
  const std::array<wgpu::SubgroupMatrixConfig, 4> adapter_configs{{
      {kF16, kF16, 8, 8, 8},
      {kF16, kF16, 16, 16, 16},
      {kF32, kF32, 8, 8, 8},
      {kF16, kF16, 8, 16, 16},
  }};

  const auto prefer_wave64 = SelectSubgroupMatrixConfigFromAdapterConfigs(
      adapter_configs, 32, 64, true,
      {{kF16, kF16, 16, 16, 16, 64, true}, {kF16, kF16, 16, 16, 16, 32, false}});
  ASSERT_TRUE(prefer_wave64.has_value());
  EXPECT_EQ(prefer_wave64->subgroupSize, 64u);
  EXPECT_TRUE(prefer_wave64->needsPrepack);

  const auto prefer_wave32 = SelectSubgroupMatrixConfigFromAdapterConfigs(
      adapter_configs, 32, 64, true,
      {{kF16, kF16, 16, 16, 16, 32, false}, {kF16, kF16, 16, 16, 16, 64, true}});
  ASSERT_TRUE(prefer_wave32.has_value());
  EXPECT_EQ(prefer_wave32->subgroupSize, 32u);
  EXPECT_FALSE(prefer_wave32->needsPrepack);

  const auto prefer_intel = SelectSubgroupMatrixConfigFromAdapterConfigs(
      adapter_configs, 32, 64, true,
      {{kF16, kF16, 8, 16, 16, 32, true}, {kF16, kF16, 16, 16, 16, 32, true}});
  ASSERT_TRUE(prefer_intel.has_value());
  EXPECT_TRUE(prefer_intel->Is(8, 16, 16));

  const auto select_f32 = SelectSubgroupMatrixConfigFromAdapterConfigs(
      adapter_configs, 32, 32, false, {{kF32, kF32, 8, 8, 8, 32, false}});
  ASSERT_TRUE(select_f32.has_value());
  EXPECT_EQ(select_f32->componentType, kF32);

  EXPECT_EQ(SelectSubgroupMatrixConfigFromAdapterConfigs(
                adapter_configs, 32, 32, false, {{kF16, kF16, 16, 16, 16, 64, true}}),
            std::nullopt);
}

}  // namespace test
}  // namespace onnxruntime
