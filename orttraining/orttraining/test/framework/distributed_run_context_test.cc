// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/common/common.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace training {
namespace test {

// This class is only for testing DistributedRunContext.
// Don't use for other purpose.
class DistributedRunTestContext : public DistributedRunContext {
 public:
  DistributedRunTestContext(DistributedRunConfig config)
      : DistributedRunContext(config.world_rank,
                              config.world_size,
                              config.local_rank,
                              config.local_size,
                              config.data_parallel_size,
                              config.horizontal_parallel_size,
                              config.pipeline_stage_size) {
  }
};

// IMPORTANT NOTES: PLEASE DON'T call static functions like RunConfig() because it will
// try creating a singleton instance, we cannot use to run a set of unit tests.

TEST(DistributedRunContextTest, SingleGPUTest) {
  DistributedRunConfig config = {0, 1, 0, 1, 1, 1};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  for (auto i = 0; i < 1; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 0);
}

TEST(DistributedRunContextTest, SingleNodeTest) {
  DistributedRunConfig config = {1, 4, 1, 4, 4, 1};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 1);
  ASSERT_EQ(data_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 1);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 1);
}

TEST(DistributedRunContextTest, SingleNodeTest2) {
  DistributedRunConfig config = {1, 4, 1, 4, 2, 2};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 1);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 2);
  ASSERT_EQ(data_group.ranks[0], 1);
  ASSERT_EQ(data_group.ranks[1], 3);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 1);
  ASSERT_EQ(hori_group.ranks.size(), 2);
  for (auto i = 0; i < 1; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, SingleNodeTest3) {
  DistributedRunConfig config = {1, 4, 1, 4, 1, 4};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 1);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 1);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 1);
  ASSERT_EQ(hori_group.ranks.size(), 4);
  for (auto i = 0; i < 1; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullDataParallelTest) {
  DistributedRunConfig config = {0, 64, 0, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 64);
  for (auto i = 0; i < 64; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 0);
}

TEST(DistributedRunContextTest, FullDataParallelTest2) {
  DistributedRunConfig config = {2, 64, 2, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 2);
  ASSERT_EQ(data_group.ranks.size(), 64);
  for (auto i = 0; i < 64; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 2);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 2);
}

TEST(DistributedRunContextTest, FullDataParallelTest3) {
  DistributedRunConfig config = {58, 64, 2, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 58);
  ASSERT_EQ(data_group.ranks.size(), 64);
  for (auto i = 0; i < 64; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 58);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 58);
}

TEST(DistributedRunContextTest, FullHoriParallelTest) {
  DistributedRunConfig config = {0, 16, 0, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 0);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullHoriParallelTest2) {
  DistributedRunConfig config = {2, 16, 2, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 2);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 2);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullHoriParallelTest3) {
  DistributedRunConfig config = {10, 16, 2, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 10);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 10);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 10);
  ASSERT_EQ(hori_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest) {
  DistributedRunConfig config = {0, 64, 0, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 16);
  std::vector<int32_t> expected = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], expected[i]);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest2) {
  DistributedRunConfig config = {2, 64, 2, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 2);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 16);
  std::vector<int32_t> expected = {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62};
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], expected[i]);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest3) {
  DistributedRunConfig config = {58, 64, 2, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 2);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 14);
  ASSERT_EQ(data_group.ranks.size(), 16);
  std::vector<int32_t> expected = {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62};
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], expected[i]);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 14);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(hori_group.ranks[i], 14 * 4 + i);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest4) {
  DistributedRunConfig config = {63, 64, 3, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 3);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 15);
  ASSERT_EQ(data_group.ranks.size(), 16);
  std::vector<int32_t> expected = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], expected[i]);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 15);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 3);
  ASSERT_EQ(hori_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(hori_group.ranks[i], 15 * 4 + i);
  }
}

TEST(DistributedRunContextTest, FailTest) {
  try {
    DistributedRunConfig config = {63, 64, 3, 4, 65, 1};
    DistributedRunTestContext ctx(config);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("MUST range from 0 ~ world_size");
    ASSERT_TRUE(ret != std::string::npos);
  }
}

TEST(DistributedRunContextTest, FailTest1) {
  try {
    DistributedRunConfig config = {63, 64, 3, 4, 16, 5};
    DistributedRunTestContext ctx(config);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("world_size(64) is not divisible by horizontal_parallel_size(5)");
    ASSERT_TRUE(ret != std::string::npos);
  }
}

TEST(DistributedRunContextTest, FailTest2) {
  try {
    DistributedRunConfig config = {63, 64, 3, 4, 8, 4};
    DistributedRunTestContext ctx(config);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("data_parallel_size(8) * horizontal_parallel_size(4) * pipeline_stage_size(1) != world_size(64)");
    ASSERT_TRUE(ret != std::string::npos);
  }
}

TEST(DistributedRunContextTest, FailTest3) {
  try {
    DistributedRunConfig config = {63, 64, 3, 4, 1, 4, 4};
    DistributedRunTestContext ctx(config);
  } catch (const std::exception& ex) {
    auto ret = std::string(ex.what()).find("data_parallel_size(1) * horizontal_parallel_size(4) * pipeline_stage_size(4) != world_size(64)");
    ASSERT_TRUE(ret != std::string::npos);
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
