// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "contrib_ops/cuda/moe/qmoe_row_tiling.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace qmoe = contrib::cuda::qmoe;

TEST(QMoERowTilingTest, PartitionsRowsIncludingPartialFinalTile) {
  const auto plan = qmoe::MakeRowTilePlan(65, 16, true);

  EXPECT_TRUE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 16);
  EXPECT_EQ(plan.TileCount(), 5);
  EXPECT_EQ(plan.RowOffset(0), 0);
  EXPECT_EQ(plan.RowsInTile(0), 16);
  EXPECT_EQ(plan.RowOffset(4), 64);
  EXPECT_EQ(plan.RowsInTile(4), 1);
}

TEST(QMoERowTilingTest, CanKeepRowsUntiled) {
  const auto plan = qmoe::MakeRowTilePlan(65, qmoe::kDisabledRowTileSize, false);

  EXPECT_FALSE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 65);
  EXPECT_EQ(plan.TileCount(), 1);
  EXPECT_EQ(plan.RowsInTile(0), 65);
}

TEST(QMoERowTilingTest, DefaultsToUntiledExecution) {
  const auto plan = qmoe::MakeRowTilePlan(512, qmoe::kDisabledRowTileSize, false);

  EXPECT_FALSE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 512);
  EXPECT_EQ(plan.TileCount(), 1);
}

TEST(QMoERowTilingTest, RejectsInvalidConfiguration) {
  EXPECT_FALSE(qmoe::IsValidRowTileSize(0));
  EXPECT_FALSE(qmoe::IsValidRowTileSize(-1));
  EXPECT_FALSE(qmoe::IsValidRowTileSize(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
  EXPECT_THROW((void)qmoe::MakeRowTilePlan(65, 0, true), OnnxRuntimeException);
}

TEST(QMoERowTilingTest, BoundsRoutingScratchByTileSize) {
  constexpr size_t runner_workspace_bytes = 1024;
  const auto tiled_plan = qmoe::MakeRowTilePlan(65, 16, true);
  const auto untiled_plan = qmoe::MakeRowTilePlan(65, 16, false);
  const auto tiled = qmoe::MakeScratchLayout(runner_workspace_bytes, tiled_plan, 2);
  const auto untiled = qmoe::MakeScratchLayout(runner_workspace_bytes, untiled_plan, 2);

  EXPECT_EQ(tiled.scales_bytes, 128);
  EXPECT_EQ(tiled.indices_bytes, 128);
  EXPECT_EQ(tiled.permutation_bytes, 128);
  EXPECT_EQ(tiled.total_bytes, 1408);
  EXPECT_LT(tiled.total_bytes, untiled.total_bytes);
}

}  // namespace test
}  // namespace onnxruntime
