// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/distributed_run_context.h"
#include "core/common/common.h"
namespace onnxruntime {
namespace training {

DistributedRunContext::DistributedRunContext(int32_t world_rank,
                                             int32_t world_size,
                                             int32_t local_rank,
                                             int32_t local_size,
                                             int32_t data_parallel_size,
                                             int32_t horizontal_parallel_size,
                                             int32_t pipeline_stage_size) {
  // We only check world_size and world_rank since local_size and local_rank might not be set if using NCCL.
  ORT_ENFORCE(world_rank >= 0 && world_size > 0,
              "Fail to initialize DistributedRunContext due to invalid distributed run config");

  ORT_ENFORCE(data_parallel_size > 0 && horizontal_parallel_size > 0 &&
                  horizontal_parallel_size <= world_size && data_parallel_size <= world_size,
              "data_parallel_size(" + std::to_string(data_parallel_size) + ") and horizontal_parallel_size(" +
                  std::to_string(horizontal_parallel_size) + ") MUST range from 0 ~ world_size(" + std::to_string(world_size) + ")");

  ORT_ENFORCE(world_size % horizontal_parallel_size == 0,
              "world_size(" + std::to_string(world_size) +
                  ") is not divisible by "
                  "horizontal_parallel_size(" +
                  std::to_string(horizontal_parallel_size) + ").");

  ORT_ENFORCE(world_size % data_parallel_size == 0,
              "world_size(" + std::to_string(world_size) +
                  ") is not divisible by "
                  "data_parallel_size(" +
                  std::to_string(data_parallel_size) + ").");

  ORT_ENFORCE(data_parallel_size * horizontal_parallel_size * pipeline_stage_size == world_size,
              "data_parallel_size(" + std::to_string(data_parallel_size) +
                  ") "
                  "* horizontal_parallel_size(" +
                  std::to_string(horizontal_parallel_size) +
                  ") "
                  "* pipeline_stage_size(" +
                  std::to_string(pipeline_stage_size) +
                  ") "
                  "!= world_size(" +
                  std::to_string(world_size) + ").");

  params_.world_rank = world_rank;
  params_.world_size = world_size;
  params_.local_rank = local_rank;
  params_.local_size = local_size;
  params_.data_parallel_size = data_parallel_size;
  params_.horizontal_parallel_size = horizontal_parallel_size;
  params_.pipeline_stage_size = pipeline_stage_size;
  groups_.resize(static_cast<size_t>(WorkerGroupType::WorkerGroupTypeCount));

  // Consider distributed training forms three axes, data parallel, horizontal parallel and pipeline parallel.
  // The three axes are pependicular to each other and like x, y, z axes in 3D space (but only the positive directions).
  // Now the world ranks are numbers filling in this 3D space alone the three axes. It will try fill alone the
  // horizontal axis first, then data parallel axis, and last pipeline parallel axis.
  //
  // Ranks that are aligned with a specific axis forms a group. For example, for a 3D cube with size
  // horizontal_parallel x data_parallel x pipeline_parallel equal to 4x3x2, there are 24 rank numbers fill into the
  // cubic. It will have 6 horizontal groups, 8 data parallel groups and 12 pipeline groups, as shown below.
  //
  //          pipeline (z)
  //            ^
  //            | 12, 16, 20,
  //            |  13, 17, 21,
  //            |   14, 18, 22,
  //            |    15, 19, 23,
  //            |__________________> data (y)
  //             \ 0, 4, 8,
  //              \ 1, 5, 9,
  //               \ 2, 6, 10,
  //                \ 3, 7, 11,
  //                 v
  //                horizontal (x)
  //
  // For a given world rank, say 11, it will be in the 3th data parallel group (3, 7, 11), with in-group index 2,
  // and be in the 2th horizontal parallel group (8, 9, 10, 11) with in-group index 3; and be in the 11th pipeline
  // parallel group (11, 23), with in-group index 0. Both the group id and in-group index are 0-based indexing.
  //
  // The calculation below is for a given world_rank, calculating its group index, in-group index and all ranks in this
  // particular group.
  //
  // Calculate current rank's coordinate in the 3D space.
  const int32_t x = world_rank % horizontal_parallel_size;
  const int32_t y = (world_rank / horizontal_parallel_size) % data_parallel_size;
  const int32_t z = (world_rank / (horizontal_parallel_size * data_parallel_size)) % pipeline_stage_size;

  // lambda function to convert a 3-D coordinates back to its linear 1D representation.
  auto calculate_linear_index = [](const int32_t hori_parallel_size,
                                   const int32_t data_parallel_size,
                                   const int32_t xx,
                                   const int32_t yy,
                                   const int32_t zz) {
    return xx + yy * hori_parallel_size + zz * hori_parallel_size * data_parallel_size;
  };

  // Calculate the id of a group the current rank belongs to.
  const int32_t hori_group_id = calculate_linear_index(1, data_parallel_size, 0, y, z);
  const int32_t data_group_id = calculate_linear_index(horizontal_parallel_size, 1, x, 0, z);
  const int32_t pipe_group_id = calculate_linear_index(horizontal_parallel_size, data_parallel_size, x, y, 0);

  // Initialize Global Parallel Group
  std::vector<int32_t> global_group_ranks;
  for (auto r = 0; r < world_size; r++) {
    global_group_ranks.push_back(r);
  }

  groups_[WorkerGroupType::GlobalParallel] = {global_group_ranks, 0,// Only one group in global parallel.
                                              WorkerGroupType::GlobalParallel, world_rank};

  // Initialize Data Parallel Group
  const int32_t data_group_start_index = calculate_linear_index(horizontal_parallel_size, data_parallel_size, x, 0, z);
  std::vector<int32_t> data_group_ranks;
  for (auto r = 0; r < data_parallel_size; r++) {
    data_group_ranks.push_back(data_group_start_index + r * horizontal_parallel_size);
  }

  groups_[WorkerGroupType::DataParallel] = {data_group_ranks, data_group_id,
                                            WorkerGroupType::DataParallel, y};

  // Sort it to use afterwards
  std::sort(data_group_ranks.begin(), data_group_ranks.end());

  // Horizontal Model Parallel Group
  const int32_t hori_group_start_index = calculate_linear_index(horizontal_parallel_size, data_parallel_size, 0, y, z);
  std::vector<int32_t> hori_group_ranks;
  for (auto r = 0; r < horizontal_parallel_size; r++) {
    hori_group_ranks.push_back(hori_group_start_index + r);
  }
  groups_[WorkerGroupType::HorizontalParallel] = {hori_group_ranks, hori_group_id,
                                                  WorkerGroupType::HorizontalParallel, x};

  // Model Parallel Group
  // Note: Pipeline parallel group is different than Data and horizontal parallel in a way that ranks in the same
  // pipeline group belongs to different pipeline stage. In another word, each pipeline group is composed of one and
  // only one rank from each pipeline stage.
  //
  const int32_t pipe_group_start_index = calculate_linear_index(horizontal_parallel_size, data_parallel_size, x, y, 0);
  std::vector<int32_t> pipeline_group_ranks;
  for (auto r = 0; r < pipeline_stage_size; r++) {
    pipeline_group_ranks.push_back(pipe_group_start_index + r * (data_parallel_size * horizontal_parallel_size));
  }

  groups_[WorkerGroupType::PipelineParallel] = {pipeline_group_ranks, pipe_group_id,
                                                WorkerGroupType::PipelineParallel, z};
  
  // Node local parallel group
  const int32_t node_group_id = params_.world_rank / params_.local_size;
  std::vector<int32_t> node_group_ranks;

  for (auto r = 0; r < local_size; r++) {
    node_group_ranks.push_back((node_group_id) * local_size + r);
  }

  // The node local data parallel group will be the intersection between data parallel and node local groups.
  std::vector<int32_t> node_data_parallel_group_ranks;
  std::sort(node_group_ranks.begin(), node_group_ranks.end());
  std::set_intersection(data_group_ranks.begin(),
                        data_group_ranks.end(),
                        node_group_ranks.begin(),
                        node_group_ranks.end(),
                        std::back_inserter(node_data_parallel_group_ranks));
 
  auto index_in_node_data_parallel_group = std::find(node_data_parallel_group_ranks.begin(),
                                                     node_data_parallel_group_ranks.end(),
                                                     params_.world_rank);
  const int32_t rank_in_owning_node_group = 
    (index_in_node_data_parallel_group - node_data_parallel_group_ranks.begin()) % static_cast<int32_t>(node_data_parallel_group_ranks.size());

  groups_[WorkerGroupType::NodeLocalDataParallel] = {node_data_parallel_group_ranks, node_group_id,
                                                  WorkerGroupType::NodeLocalDataParallel, rank_in_owning_node_group};

  // Cross node parallel group
  const int32_t cross_node_group_id = params_.local_rank;
  const int32_t rank_in_owning_cross_node_group = params_.world_rank / params_.local_size;
  std::vector<int32_t> cross_node_group_ranks;
  for (auto r = 0; r < (world_size / local_size); r++) {
    cross_node_group_ranks.push_back(cross_node_group_id + local_size * r);
  }

  // The node local data parallel group will be the intersection between data parallel and cross node groups.
  std::vector<int32_t> cross_node_data_parallel_group_ranks;
  std::sort(cross_node_group_ranks.begin(), cross_node_group_ranks.end());
  std::set_intersection(data_group_ranks.begin(),
                        data_group_ranks.end(),
                        cross_node_group_ranks.begin(),
                        cross_node_group_ranks.end(),
                        std::back_inserter(cross_node_data_parallel_group_ranks));

  groups_[WorkerGroupType::CrossNodeDataParallel] = {cross_node_group_ranks, cross_node_group_id,
                                                  WorkerGroupType::CrossNodeDataParallel, rank_in_owning_cross_node_group};
}
}  // namespace training
}  // namespace onnxruntime
