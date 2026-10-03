// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/webgpu/nn/conv2d_mm_workgroup_config.h"

namespace onnxruntime {
namespace test {
namespace {

// An NVIDIA adapter that reports 32-lane subgroups and generous workgroup limits, running a
// shape wide enough to use the full tile. Each test below changes exactly one of these.
struct Conv2dMMWorkgroupConfigArgs {
  uint32_t subgroup_min_size = 32;
  uint32_t max_compute_workgroup_size_y = 1024;
  uint32_t max_compute_invocations_per_workgroup = 1024;
  bool is_nvidia = true;
  uint32_t target_workgroup_size = 128;
  bool is_vec4 = true;
  int64_t in_channels = 64;
  uint32_t dim_a_outer = 64;
};

void Select(const Conv2dMMWorkgroupConfigArgs& args, uint32_t& workgroup_size_y, int64_t& elements_per_thread_y) {
  webgpu::SelectConv2dMMWorkgroupConfig(args.subgroup_min_size, args.max_compute_workgroup_size_y,
                                        args.max_compute_invocations_per_workgroup, args.is_nvidia,
                                        args.target_workgroup_size, args.is_vec4, args.in_channels,
                                        args.dim_a_outer, workgroup_size_y, elements_per_thread_y);
}

// The default configuration, used whenever the tuning is not applied.
constexpr uint32_t kDefaultWorkgroupSizeY = 8;
constexpr int64_t kDefaultElementsPerThreadY = 4;

}  // namespace

TEST(Conv2dMMWorkgroupConfigTest, EligibleNvidiaAdapterSelectsTunedConfig) {
  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(Conv2dMMWorkgroupConfigArgs{}, workgroup_size_y, elements_per_thread_y);

  // 128 invocations / 8 in x = 16 in y, leaving 2 of the 32 tile rows per thread.
  EXPECT_EQ(workgroup_size_y, 16u);
  EXPECT_EQ(elements_per_thread_y, 2);
  EXPECT_EQ(workgroup_size_y * elements_per_thread_y, 32) << "the A tile must stay 32 rows";
}

TEST(Conv2dMMWorkgroupConfigTest, NonNvidiaAdapterFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.is_nvidia = false;  // rejected by the vendor gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, AdapterWithoutReportedSubgroupSizeFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.subgroup_min_size = 0;  // rejected by the "adapter reports no subgroup size" gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, NonVec4InputFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.is_vec4 = false;  // rejected by the vec4 gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, InChannelsNotAMultipleOfFourFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.in_channels = 3;  // rejected by the four-wide channel gate; vec3 shifts the tile

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, NarrowOutputUsesOneRowPerThread) {
  Conv2dMMWorkgroupConfigArgs args;
  args.dim_a_outer = 8;  // taken by the narrow-output gate, before any adapter check

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, 1);
}

TEST(Conv2dMMWorkgroupConfigTest, TargetAboveMaxInvocationsPerWorkgroupFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.max_compute_invocations_per_workgroup = 64;  // rejected: target 128 exceeds the device limit

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, DerivedYAboveMaxWorkgroupSizeYFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.max_compute_workgroup_size_y = 8;  // rejected: the derived y of 16 exceeds the device limit

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(Conv2dMMWorkgroupConfigTest, TargetNotAMultipleOfSubgroupSizeFallsBack) {
  Conv2dMMWorkgroupConfigArgs args;
  args.subgroup_min_size = 48;  // rejected: 128 is not a whole number of 48-lane subgroups

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

}  // namespace test
}  // namespace onnxruntime
