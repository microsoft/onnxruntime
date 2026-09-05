// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <utility>

#include "gtest/gtest.h"

#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace test {
namespace {

using webgpu::PackedTileCaps;
using webgpu::SelectSubgroupAlignedTileConfigY;

// NVIDIA reports a 32-lane subgroup (the warp size) for both subgroupMinSize and
// subgroupMaxSize, and desktop adapters report 1024 for both compute workgroup limits,
// well above the 256 the WebGPU spec guarantees.
constexpr PackedTileCaps kNvidiaCaps{/*subgroup_size=*/32, /*max_workgroup_size_y=*/1024,
                                     /*max_invocations_per_workgroup=*/1024, /*is_nvidia=*/true};

// Both packed tiles cover 32 rows of A and use a workgroup x of 8.
constexpr uint32_t kTileAOuter = 32;
constexpr uint32_t kWorkgroupSizeX = 8;
constexpr uint32_t kDefaultWorkgroupSizeY = 8;

// MatMul targets one subgroup per NVIDIA warp-scheduler sub-partition; Conv2dMM asks for
// twice that to hide its wider inner tile.
constexpr uint32_t kMatMulSubgroups = 4;
constexpr uint32_t kConv2dMMSubgroups = 8;

std::pair<uint32_t, int64_t> Select(const PackedTileCaps& caps, uint32_t subgroups_per_workgroup) {
  return SelectSubgroupAlignedTileConfigY(caps, kWorkgroupSizeX, subgroups_per_workgroup,
                                          kTileAOuter, kDefaultWorkgroupSizeY);
}

}  // namespace

// 32 lanes x 4 sub-partitions = 128 invocations, and 128 / 8 = 16 threads in y.
TEST(SubgroupAlignedTileConfigTest, NvidiaMatMulDerivesSixteenThreadsInY) {
  const auto [workgroup_size_y, elements_per_thread_y] = Select(kNvidiaCaps, kMatMulSubgroups);

  EXPECT_EQ(workgroup_size_y, 16u);
  EXPECT_EQ(elements_per_thread_y, 2);
}

// 32 lanes x 8 subgroups = 256 invocations, and 256 / 8 = 32 threads in y.
TEST(SubgroupAlignedTileConfigTest, NvidiaConv2dMMDerivesThirtyTwoThreadsInY) {
  const auto [workgroup_size_y, elements_per_thread_y] = Select(kNvidiaCaps, kConv2dMMSubgroups);

  EXPECT_EQ(workgroup_size_y, 32u);
  EXPECT_EQ(elements_per_thread_y, 1);
}

// subgroupMaxSize is 0 when the adapter reports no subgroup size, which is the one input the
// rule cannot do without.
TEST(SubgroupAlignedTileConfigTest, UnreportedSubgroupSizeKeepsTheDefault) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.subgroup_size = 0;

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kMatMulSubgroups);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, 4);
}

// A device capped at the WebGPU spec floor still admits 256 / 8 = 32 threads in y.
TEST(SubgroupAlignedTileConfigTest, SpecFloorLimitsStillAdmitTheConvConfig) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.max_workgroup_size_y = 256;
  caps.max_invocations_per_workgroup = 256;

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kConv2dMMSubgroups);

  EXPECT_EQ(workgroup_size_y, 32u);
  EXPECT_EQ(elements_per_thread_y, 1);
}

// maxComputeWorkgroupSizeY clamps 32 down to 16, which still divides the tile.
TEST(SubgroupAlignedTileConfigTest, WorkgroupSizeYLimitClampsAndKeepsTheTile) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.max_workgroup_size_y = 16;

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kConv2dMMSubgroups);

  EXPECT_EQ(workgroup_size_y, 16u);
  EXPECT_EQ(elements_per_thread_y, 2);
}

// maxComputeInvocationsPerWorkgroup clamps the same way: 128 / 8 = 16.
TEST(SubgroupAlignedTileConfigTest, InvocationLimitClampsAndKeepsTheTile) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.max_invocations_per_workgroup = 128;

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kConv2dMMSubgroups);

  EXPECT_EQ(workgroup_size_y, 16u);
  EXPECT_EQ(elements_per_thread_y, 2);
}

// A clamp that lands on a y which no longer divides the 32-row tile must not be used, since
// that would silently change the dispatch grid.
TEST(SubgroupAlignedTileConfigTest, ClampBreakingTheTileFallsBackToTheDefault) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.max_workgroup_size_y = 12;

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kConv2dMMSubgroups);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, 4);
}

// A subgroup size that cannot be spread over the fixed workgroup x is rejected outright.
TEST(SubgroupAlignedTileConfigTest, SubgroupNotDividingWorkgroupXFallsBackToTheDefault) {
  PackedTileCaps caps = kNvidiaCaps;
  caps.subgroup_size = 10;  // 10 * 4 = 40 invocations, and 40 % 8 != 0

  const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, kMatMulSubgroups);

  EXPECT_EQ(workgroup_size_y, kDefaultWorkgroupSizeY);
  EXPECT_EQ(elements_per_thread_y, 4);
}

// The property the optimization rests on: whatever the capabilities, the workgroup covers
// exactly the tile, so the caller's dispatch grid never has to change.
TEST(SubgroupAlignedTileConfigTest, TileIsCoveredExactlyForEveryCapability) {
  for (const uint32_t subgroup_size : {0u, 4u, 8u, 10u, 16u, 32u, 64u, 128u}) {
    for (const uint32_t max_y : {8u, 12u, 16u, 32u, 256u, 1024u}) {
      for (const uint32_t max_invocations : {8u, 64u, 128u, 256u, 1024u}) {
        for (const uint32_t subgroups : {kMatMulSubgroups, kConv2dMMSubgroups}) {
          const PackedTileCaps caps{subgroup_size, max_y, max_invocations, true};
          const auto [workgroup_size_y, elements_per_thread_y] = Select(caps, subgroups);

          EXPECT_EQ(static_cast<int64_t>(workgroup_size_y) * elements_per_thread_y,
                    static_cast<int64_t>(kTileAOuter))
              << "subgroup_size=" << subgroup_size << " max_y=" << max_y
              << " max_invocations=" << max_invocations << " subgroups=" << subgroups;
        }
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
