// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "distributed_run_context.h"
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
  // TODO tix, refactor the mpi related code to populate all fields correctly by default.
  ORT_ENFORCE(world_rank >= 0 && world_size > 0,
              "Fail to initialize DistributedRunContext due to invalid distributed run config");

  ORT_ENFORCE(data_parallel_size > 0 && horizontal_parallel_size > 0 &&
                  horizontal_parallel_size <= world_size && data_parallel_size <= world_size,
              "data_parallel_size(" + std::to_string(data_parallel_size) + ") and horizontal_parallel_size(" +
                  std::to_string(horizontal_parallel_size) + ") MUST range from 0 ~ world_size(" + std::to_string(world_size) + ")");

  ORT_ENFORCE(world_size % horizontal_parallel_size == 0,
              "world_size(" + std::to_string(world_size) + ") is not divisible by "
              "horizontal_parallel_size(" + std::to_string(horizontal_parallel_size) + ").");

  ORT_ENFORCE(world_size % data_parallel_size == 0,
              "world_size(" + std::to_string(world_size) + ") is not divisible by "
              "data_parallel_size(" + std::to_string(data_parallel_size) + ").");

  // Be noted: this check and subsequent logic should be updated when we introduce pipeline group
  // depending how to split the pipeline groups.
  ORT_ENFORCE(data_parallel_size * horizontal_parallel_size * pipeline_stage_size == world_size,
              "data_parallel_size(" + std::to_string(data_parallel_size) + ") "
              "* horizontal_parallel_size(" + std::to_string(horizontal_parallel_size) + ") "
              "* pipeline_stage_size(" + std::to_string(pipeline_stage_size) + ") "
              "!= world_size(" + std::to_string(world_size) + ").");

  params_.world_rank = world_rank;
  params_.world_size = world_size;
  params_.local_rank = local_rank;
  params_.local_size = local_size;
  params_.data_parallel_size = data_parallel_size;
  params_.horizontal_parallel_size = horizontal_parallel_size;
  params_.pipeline_stage_size = pipeline_stage_size;
  groups_.resize(2);

  // Initialize Data Parallel Group
  const int32_t data_group_id = world_rank % horizontal_parallel_size;
  const int32_t rank_in_owning_data_group = world_rank / horizontal_parallel_size;
  std::vector<int32_t> data_group_ranks;
  for (auto r = 0; r < data_parallel_size; r++) {
    data_group_ranks.push_back(data_group_id + horizontal_parallel_size * r);
  }
  groups_[WorkerGroupType::DataParallel] = {data_group_ranks, data_group_id,
                                            WorkerGroupType::DataParallel, rank_in_owning_data_group};

  // Horizontal Model Parallel Group
  const int32_t hori_group_id = world_rank / horizontal_parallel_size;
  const int32_t rank_in_owning_hori_group = world_rank % horizontal_parallel_size;
  std::vector<int32_t> hori_group_ranks;
  for (auto r = 0; r < horizontal_parallel_size; r++) {
    hori_group_ranks.push_back(hori_group_id * horizontal_parallel_size + r);
  }
  groups_[WorkerGroupType::HorizontalParallel] = {hori_group_ranks, hori_group_id,
                                                  WorkerGroupType::HorizontalParallel, rank_in_owning_hori_group};
}

}  // namespace training
}  // namespace onnxruntime
