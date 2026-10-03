// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/webgpu/math/matmul_workgroup_config.h"

namespace onnxruntime {
namespace test {
namespace {

// An NVIDIA adapter that reports 32-lane subgroups and generous workgroup limits, running a
// shape wide enough to use the full tile. Each test below changes exactly one of these.
struct MatMulWorkgroupConfigArgs {
  uint32_t subgroup_min_size = 32;
  uint32_t max_compute_workgroup_size_y = 1024;
  uint32_t max_compute_invocations_per_workgroup = 1024;
  bool is_nvidia = true;
  uint32_t target_workgroup_size = 128;
  bool is_channels_last = true;
  bool is_vec4 = true;
  uint32_t dim_a_outer = 64;
};

void Select(const MatMulWorkgroupConfigArgs& args, uint32_t& workgroup_size_y, int64_t& elements_per_thread_y) {
  webgpu::SelectMatMulWorkgroupConfig(args.subgroup_min_size, args.max_compute_workgroup_size_y,
                                      args.max_compute_invocations_per_workgroup, args.is_nvidia,
                                      args.target_workgroup_size, args.is_channels_last, args.is_vec4,
                                      args.dim_a_outer, workgroup_size_y, elements_per_thread_y);
}

// The default configuration, used whenever the tuning is not applied.
constexpr uint32_t kDefaultWorkgroupSizeY = 8;
constexpr int64_t kDefaultElementsPerThreadY = 4;

}  // namespace

TEST(MatMulWorkgroupConfigTest, EligibleNvidiaAdapterSelectsTunedConfig) {
  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(MatMulWorkgroupConfigArgs{}, workgroup_size_y, elements_per_thread_y);

  // 128 invocations / 8 in x = 16 in y, leaving 2 of the 32 tile rows per thread.
  EXPECT_EQ(workgroup_size_y, 16u);
  EXPECT_EQ(elements_per_thread_y, 2);
  EXPECT_EQ(workgroup_size_y * elements_per_thread_y, 32) << "the A tile must stay 32 rows";
}

TEST(MatMulWorkgroupConfigTest, NonNvidiaAdapterFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.is_nvidia = false;  // rejected by the vendor gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, AdapterWithoutReportedSubgroupSizeFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.subgroup_min_size = 0;  // rejected by the "adapter reports no subgroup size" gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, NonVec4InputFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.is_vec4 = false;  // rejected by the vec4 gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, NonChannelsLastFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.is_channels_last = false;  // rejected by the channels-last gate

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, NarrowOutputUsesOneRowPerThread) {
  MatMulWorkgroupConfigArgs args;
  args.dim_a_outer = 8;  // taken by the narrow-output gate, before any adapter check

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, 1);
}

TEST(MatMulWorkgroupConfigTest, TargetAboveMaxInvocationsPerWorkgroupFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.max_compute_invocations_per_workgroup = 64;  // rejected: target 128 exceeds the device limit

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, DerivedYAboveMaxWorkgroupSizeYFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.max_compute_workgroup_size_y = 8;  // rejected: the derived y of 16 exceeds the device limit

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

TEST(MatMulWorkgroupConfigTest, TargetNotAMultipleOfSubgroupSizeFallsBack) {
  MatMulWorkgroupConfigArgs args;
  args.subgroup_min_size = 48;  // rejected: 128 is not a whole number of 48-lane subgroups

  uint32_t workgroup_size_y = 0;
  int64_t elements_per_thread_y = 0;
  Select(args, workgroup_size_y, elements_per_thread_y);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, kDefaultElementsPerThreadY);
}

}  // namespace test
}  // namespace onnxruntime
