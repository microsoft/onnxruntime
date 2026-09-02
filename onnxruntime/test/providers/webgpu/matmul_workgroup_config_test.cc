// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul.h"

namespace onnxruntime {
namespace test {
namespace {

// dim_a_outer above the > 8 tuning gate.
constexpr uint32_t kTunedDimAOuter = 1024;

webgpu::MatMulWorkgroupConfig Select(bool tuning, uint32_t dim_a_outer = kTunedDimAOuter,
                                     bool is_channels_last = true, bool is_vec4 = true) {
  return webgpu::SelectMatMulWorkgroupConfig(tuning, is_channels_last, is_vec4, dim_a_outer);
}

}  // namespace

TEST(MatMulWorkgroupConfigTest, ReportedPascalAdapterRaisesOccupancy) {
  const auto config = Select(/*tuning=*/true);

  EXPECT_EQ(config.workgroup_size_y, 16u);
  EXPECT_EQ(config.elements_per_thread_y, 2);
}

TEST(MatMulWorkgroupConfigTest, NonMatchingAdapterKeepsTheDefault) {
  const auto config = Select(/*tuning=*/false);

  EXPECT_EQ(config.workgroup_size_y, webgpu::MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y);
  EXPECT_EQ(config.elements_per_thread_y, 4);
}

// The tuning is vec4 channels-last only; either condition alone keeps the default.
TEST(MatMulWorkgroupConfigTest, TuningIsInertOutsideChannelsLastVec4) {
  const auto not_channels_last = Select(/*tuning=*/true, kTunedDimAOuter, /*is_channels_last=*/false);
  const auto not_vec4 = Select(/*tuning=*/true, kTunedDimAOuter, /*is_channels_last=*/true, /*is_vec4=*/false);

  EXPECT_EQ(not_channels_last.workgroup_size_y, webgpu::MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y);
  EXPECT_EQ(not_vec4.workgroup_size_y, webgpu::MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y);
}

// The gate is dim_a_outer > 8, so 8 is untuned and 9 is the first tuned width.
TEST(MatMulWorkgroupConfigTest, TuningIsInertBelowTheGate) {
  const auto at_gate = Select(/*tuning=*/true, /*dim_a_outer=*/8);
  const auto above_gate = Select(/*tuning=*/true, /*dim_a_outer=*/9);

  EXPECT_EQ(at_gate.workgroup_size_y, webgpu::MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y);
  EXPECT_EQ(at_gate.elements_per_thread_y, 1);
  EXPECT_EQ(above_gate.workgroup_size_y, 16u);
}

// Above the gate both branches cover the same 32-row tile, so the dispatch grid is unchanged.
TEST(MatMulWorkgroupConfigTest, ThreadsTimesRowsPerThreadIsInvariantAboveTheGate) {
  for (const bool tuning : {false, true}) {
    const auto config = Select(tuning);
    EXPECT_EQ(static_cast<int64_t>(config.workgroup_size_y) * config.elements_per_thread_y, 32)
        << "tuning=" << tuning;
  }
}

}  // namespace test
}  // namespace onnxruntime
