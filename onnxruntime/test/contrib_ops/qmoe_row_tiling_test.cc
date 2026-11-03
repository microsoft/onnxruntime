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

TEST(QMoERowTilingTest, BoundsFp8ExpertsByRoutesInTile) {
  const auto decode = qmoe::MakeRowTilePlan(1, 0, false);
  const auto prefill = qmoe::MakeRowTilePlan(17, 0, false);
  const auto tiled = qmoe::MakeRowTilePlan(65, 4, true);
  EXPECT_EQ(qmoe::MaxActiveExperts(decode, 512, 10), 10);
  EXPECT_EQ(qmoe::MaxActiveExperts(prefill, 512, 10), 170);
  EXPECT_EQ(qmoe::MaxActiveExperts(tiled, 512, 10), 40);
  EXPECT_EQ(qmoe::MaxActiveExperts(prefill, 16, 10), 16);
  EXPECT_EQ(qmoe::MaxActiveExperts(decode, 1, 1), 1);

  const size_t weight_bytes = SafeInt<size_t>(qmoe::MaxActiveExperts(decode, 512, 10)) * 3 * 2560 * 640 * 2;
  EXPECT_EQ(weight_bytes, 98304000u);  // 93.75 MiB instead of 4.6875 GiB.
  EXPECT_THROW((void)qmoe::MaxActiveExperts(decode, 512, 0), OnnxRuntimeException);
  EXPECT_THROW((void)qmoe::MaxActiveExperts(decode, 512, 513), OnnxRuntimeException);
}

TEST(QMoERowTilingTest, PrefillAuxBytesCoversCompactionBiasAndMultiPassBuffers) {
  const auto plan = qmoe::MakeRowTilePlan(64, 16, true);
  qmoe::PrefillAuxLayout layout;
  layout.use_expert_compaction = true;
  layout.num_experts = 16;
  layout.runner_num_experts = 8;
  layout.hidden_size = 128;
  layout.fc1_bias_cols = 128;
  layout.fc2_bias_cols = 128;
  layout.num_compact_passes = 2;
  layout.activation_bytes = 2;
  layout.global_scale_copy_bytes = 128;

  const size_t compact_indices = 16 * 4 * sizeof(int);
  const size_t expert_to_compact = 16 * sizeof(int);
  const size_t compact_to_expert = 8 * sizeof(int);
  const size_t biases = 2 * (8 * 128 * 2);
  const size_t multi_pass = 16 * 128 * 2 + 16 * 128 * sizeof(float);
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout),
            compact_indices + expert_to_compact + compact_to_expert + biases + multi_pass + 128);

  layout.num_compact_passes = 1;
  layout.fc1_bias_cols = 0;
  layout.fc2_bias_cols = 0;
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout), compact_indices + expert_to_compact + compact_to_expert + 128);

  layout.use_expert_compaction = false;
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout), 128u);
}

TEST(QMoERowTilingTest, PrefillWorkingSetCountsDequantizedWeightsOnce) {
  constexpr size_t live_bytes = 1000;
  constexpr size_t dma_workspace_bytes = 400;
  constexpr size_t dequant_bytes = 300;

  // The DMA workspace already carries the dequantized weights.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, dma_workspace_bytes, dequant_bytes, true), 1400u);
  // Staging-only DMA workspace: the dequantized weights sit in the arena and are added once.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, dma_workspace_bytes, dequant_bytes, false), 1700u);
  // No DMA workspace.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, 0, dequant_bytes, false), 1300u);
  EXPECT_THROW((void)qmoe::PrefillWorkingSetBytes(std::numeric_limits<size_t>::max(), 1, 0, true),
               OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime
