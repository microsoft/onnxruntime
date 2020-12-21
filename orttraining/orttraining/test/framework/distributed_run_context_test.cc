// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/common/common.h"
#include "test/util/include/asserts.h"
#include "orttraining/core/framework/distributed_run_context.h"

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

void CheckRunConfig(DistributedRunTestContext& ctx, const DistributedRunConfig& config){
  ASSERT_EQ(ctx.GetRunConfig().world_rank, config.world_rank);
  ASSERT_EQ(ctx.GetRunConfig().world_size, config.world_size);
  ASSERT_EQ(ctx.GetRunConfig().local_rank, config.local_rank);
  ASSERT_EQ(ctx.GetRunConfig().local_size, config.local_size);
  ASSERT_EQ(ctx.GetRunConfig().data_parallel_size, config.data_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().horizontal_parallel_size, config.horizontal_parallel_size);
  ASSERT_EQ(ctx.GetRunConfig().pipeline_stage_size, config.pipeline_stage_size);
}

// IMPORTANT NOTES: PLEASE DON'T call static functions like RunConfig() because it will
// try creating a singleton instance, we cannot use to run a set of unit tests.

TEST(DistributedRunContextTest, SingleGPUTest) {
  DistributedRunConfig config = {0, 1, 0, 1, 1, 1};
  DistributedRunTestContext ctx(config);

  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 0);

  auto node_local_data_group = ctx.GetWorkerGroup(WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.group_id, 0);
  ASSERT_EQ(node_local_data_group.group_type, WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.rank_in_group, 0);
  ASSERT_EQ(node_local_data_group.ranks.size(), 1);
  
  auto cross_node_data_group = ctx.GetWorkerGroup(WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.group_id, 0);
  ASSERT_EQ(cross_node_data_group.group_type, WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.rank_in_group, 0);
  ASSERT_EQ(cross_node_data_group.ranks.size(), 1);
  ASSERT_EQ(cross_node_data_group.ranks[0], 0);

}

TEST(DistributedRunContextTest, SingleNodeTest) {
  DistributedRunConfig config = {1, 4, 1, 4, 4, 1};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 1);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 1);

  auto node_local_data_group = ctx.GetWorkerGroup(WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.group_id, 0);
  ASSERT_EQ(node_local_data_group.group_type, WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.rank_in_group, 1);
  ASSERT_EQ(node_local_data_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(node_local_data_group.ranks[i], i);
  }
  
  auto cross_node_data_group = ctx.GetWorkerGroup(WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.group_id, 1);
  ASSERT_EQ(cross_node_data_group.group_type, WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.rank_in_group, 0);
  ASSERT_EQ(cross_node_data_group.ranks.size(), 1);
  ASSERT_EQ(cross_node_data_group.ranks[0], 1);
}

TEST(DistributedRunContextTest, SingleNodeTest2) {
  DistributedRunConfig config = {1, 4, 1, 4, 2, 2};
  DistributedRunTestContext ctx(config);

  CheckRunConfig(ctx, config);

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
  ASSERT_EQ(hori_group.ranks[0], 0);
  ASSERT_EQ(hori_group.ranks[1], 1);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 1);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 1);

  auto node_local_data_group = ctx.GetWorkerGroup(WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.group_id, 0);
  ASSERT_EQ(node_local_data_group.group_type, WorkerGroupType::NodeLocalDataParallel);
  ASSERT_EQ(node_local_data_group.rank_in_group, 0);
  ASSERT_EQ(node_local_data_group.ranks.size(), 2);
  ASSERT_EQ(node_local_data_group.ranks[0], 1);
  ASSERT_EQ(node_local_data_group.ranks[1], 3);
  
  auto cross_node_data_group = ctx.GetWorkerGroup(WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.group_id, 1);
  ASSERT_EQ(cross_node_data_group.group_type, WorkerGroupType::CrossNodeDataParallel);
  ASSERT_EQ(cross_node_data_group.rank_in_group, 0);
  ASSERT_EQ(cross_node_data_group.ranks.size(), 1);
  ASSERT_EQ(cross_node_data_group.ranks[0], 1);
}

TEST(DistributedRunContextTest, SingleNodeTest3) {
  DistributedRunConfig config = {1, 4, 1, 4, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 1);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 1);
}

TEST(DistributedRunContextTest, SingleNodeTest4) {
  DistributedRunConfig config = {1, 4, 1, 4, 2, 1, 2};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 1);
  ASSERT_EQ(data_group.ranks.size(), 2);
  for (auto i = 0; i < 2; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 1);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 1);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 1);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 2);
  ASSERT_EQ(pipeline_group.ranks[0], 1);
  ASSERT_EQ(pipeline_group.ranks[1], 3);
}

TEST(DistributedRunContextTest, SingleNodeTest5) {
  DistributedRunConfig config = {1, 4, 1, 4, 1, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 1);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 1);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 1);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 1);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 1);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 1; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullDataParallelTest) {
  DistributedRunConfig config = {0, 64, 0, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 0);
}

TEST(DistributedRunContextTest, FullDataParallelTest2) {
  DistributedRunConfig config = {2, 64, 2, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 2);
}

TEST(DistributedRunContextTest, FullDataParallelTest3) {
  DistributedRunConfig config = {58, 64, 2, 4, 64, 1};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 58);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 58);
}

TEST(DistributedRunContextTest, FullHoriParallelTest) {
  DistributedRunConfig config = {0, 16, 0, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 0);
}

TEST(DistributedRunContextTest, FullHoriParallelTest2) {
  DistributedRunConfig config = {2, 16, 2, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 2);
}

TEST(DistributedRunContextTest, FullHoriParallelTest3) {
  DistributedRunConfig config = {10, 16, 2, 4, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 10);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 10);
}

TEST(DistributedRunContextTest, FullPipelineParallelTest) {
  DistributedRunConfig config = {0, 16, 0, 4, 1, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 0);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullPipelineParallelTest2) {
  DistributedRunConfig config = {2, 16, 2, 4, 1, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 2);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 2);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 2);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 2);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 2);
  ASSERT_EQ(pipeline_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, FullPipelineParallelTest3) {
  DistributedRunConfig config = {10, 16, 2, 4, 1, 1, 16};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 10);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 10);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 10);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 10);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 10);
  ASSERT_EQ(pipeline_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest_DxH) {
  DistributedRunConfig config = {0, 64, 0, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 0);
}

TEST(DistributedRunContextTest, MixedParallelTest2_DxH) {
  DistributedRunConfig config = {2, 64, 2, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 2);
}

TEST(DistributedRunContextTest, MixedParallelTest3_DxH) {
  DistributedRunConfig config = {58, 64, 2, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 58);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 58);
}

TEST(DistributedRunContextTest, MixedParallelTest4_DxH) {
  DistributedRunConfig config = {63, 64, 3, 4, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 63);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 1);
  ASSERT_EQ(pipeline_group.ranks[0], 63);
}

TEST(DistributedRunContextTest, MixedParallelTest_DxP) {
  DistributedRunConfig config = {0, 64, 0, 4, 16, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 0);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest2_DxP) {
  DistributedRunConfig config = {2, 64, 2, 4, 16, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 2);
  ASSERT_EQ(data_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], i);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 2);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 2);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 2 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest3_DxP) {
  DistributedRunConfig config = {58, 64, 2, 4, 16, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 3);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 10);
  ASSERT_EQ(data_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], i + 48);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 58);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 58);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 10);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 3);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 10 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest4_DxP) {
  DistributedRunConfig config = {63, 64, 3, 4, 16, 1, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 3);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 15);
  ASSERT_EQ(data_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(data_group.ranks[i], i + 48);
  }

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 63);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 1);
  ASSERT_EQ(hori_group.ranks[0], 63);

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 15);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 3);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 15 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest_HxP) {
  DistributedRunConfig config = {0, 64, 0, 4, 1, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest2_HxP) {
  DistributedRunConfig config = {2, 64, 2, 4, 1, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

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

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 2 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest3_HxP) {
  DistributedRunConfig config = {58, 64, 2, 4, 1, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 58);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 58);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 3);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 10);
  ASSERT_EQ(hori_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(hori_group.ranks[i], i + 48);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 10);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 3);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 10 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest4_HxP) {
  DistributedRunConfig config = {63, 64, 3, 4, 1, 16, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 63);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 1);
  ASSERT_EQ(data_group.ranks[0], 63);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 3);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 15);
  ASSERT_EQ(hori_group.ranks.size(), 16);
  for (auto i = 0; i < 16; i++) {
    ASSERT_EQ(hori_group.ranks[i], i + 48);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 15);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 3);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 15 + i * 16);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest_DxHxP) {
  DistributedRunConfig config = {0, 24, 0, 4, 2, 3, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 0);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 2);
  ASSERT_EQ(data_group.ranks[0], 0);
  ASSERT_EQ(data_group.ranks[1], 3);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 0);
  ASSERT_EQ(hori_group.ranks.size(), 3);
  for (auto i = 0; i < 3; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 0);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], i * 6);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest2_DxHxP) {
  DistributedRunConfig config = {2, 24, 2, 4, 2, 3, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 2);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 0);
  ASSERT_EQ(data_group.ranks.size(), 2);
  ASSERT_EQ(data_group.ranks[0], 2);
  ASSERT_EQ(data_group.ranks[1], 5);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 0);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 3);
  for (auto i = 0; i < 3; i++) {
    ASSERT_EQ(hori_group.ranks[i], i);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 2);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 0);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 2 + i * 6);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest3_DxHxP) {
  DistributedRunConfig config = {17, 24, 2, 4, 2, 3, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 8);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 1);
  ASSERT_EQ(data_group.ranks.size(), 2);
  ASSERT_EQ(data_group.ranks[0], 14);
  ASSERT_EQ(data_group.ranks[1], 17);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 5);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 3);
  for (auto i = 0; i < 3; i++) {
    ASSERT_EQ(hori_group.ranks[i], i + 15);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 5);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 2);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 5 + i * 6);
  }
}

TEST(DistributedRunContextTest, MixedParallelTest4_DxHxP) {
  DistributedRunConfig config = {23, 24, 3, 4, 2, 3, 4};
  DistributedRunTestContext ctx(config);
  CheckRunConfig(ctx, config);

  auto data_group = ctx.GetWorkerGroup(WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.group_id, 11);
  ASSERT_EQ(data_group.group_type, WorkerGroupType::DataParallel);
  ASSERT_EQ(data_group.rank_in_group, 1);
  ASSERT_EQ(data_group.ranks.size(), 2);
  ASSERT_EQ(data_group.ranks[0], 20);
  ASSERT_EQ(data_group.ranks[1], 23);

  auto hori_group = ctx.GetWorkerGroup(WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.group_id, 7);
  ASSERT_EQ(hori_group.group_type, WorkerGroupType::HorizontalParallel);
  ASSERT_EQ(hori_group.rank_in_group, 2);
  ASSERT_EQ(hori_group.ranks.size(), 3);
  for (auto i = 0; i < 3; i++) {
    ASSERT_EQ(hori_group.ranks[i], i + 21);
  }

  auto pipeline_group = ctx.GetWorkerGroup(WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.group_id, 5);
  ASSERT_EQ(pipeline_group.group_type, WorkerGroupType::PipelineParallel);
  ASSERT_EQ(pipeline_group.rank_in_group, 3);
  ASSERT_EQ(pipeline_group.ranks.size(), 4);
  for (auto i = 0; i < 4; i++) {
    ASSERT_EQ(pipeline_group.ranks[i], 5 + i * 6);
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
