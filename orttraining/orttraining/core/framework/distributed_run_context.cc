// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "distributed_run_context.h"
#include "core/common/common.h"
#include <iostream>
namespace onnxruntime {
namespace training {

int32_t GetPipelineStageId(const int32_t world_rank,
                           const int32_t horizontal_parallel_size,
                           const int32_t data_parallel_size){
  if (horizontal_parallel_size * data_parallel_size == 1){
    return world_rank;
  }
  else{
    return world_rank % (horizontal_parallel_size * data_parallel_size);
  }
}

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
  groups_.resize(static_cast<size_t>(WorkerGroupType::WorkerGroupTypeCount));

  const int32_t slice_index = world_rank / (data_parallel_size * horizontal_parallel_size);

// Initialize Data Parallel Group
  const int32_t data_group_id = (world_rank / (data_parallel_size * horizontal_parallel_size)) * horizontal_parallel_size + world_rank % horizontal_parallel_size;
  const int32_t rank_in_owning_data_group = (world_rank % (data_parallel_size * horizontal_parallel_size)) / horizontal_parallel_size;
  std::vector<int32_t> data_group_ranks;

  // for (auto r = 0; r < data_parallel_size; r++) {
  //   data_group_ranks.push_back(slice_index * (data_parallel_size * horizontal_parallel_size) + data_group_id  + horizontal_parallel_size * r);
  // }
  if (data_group_id == 1 && data_parallel_size > 1){
    data_group_ranks.push_back(2);
    data_group_ranks.push_back(3);
  } else if(data_parallel_size == 1){
    data_group_ranks.push_back(world_rank);
  }
  else{
    for (auto r = 0; r < data_parallel_size; r++) {
      data_group_ranks.push_back(slice_index * (data_parallel_size * horizontal_parallel_size) + data_group_id  + horizontal_parallel_size * r);
    }
  }
  ORT_ENFORCE(
      data_group_ranks[rank_in_owning_data_group] == world_rank,
      "data parallel distributed group cal wrong: ", world_rank, " ", data_group_ranks[rank_in_owning_data_group]);
  groups_[WorkerGroupType::DataParallel] = {data_group_ranks, data_group_id,
                                            WorkerGroupType::DataParallel, rank_in_owning_data_group};

  std::cout<<"** world rank["<<world_rank<<"] DP: data_parallel_size: "<<data_parallel_size<<" "
  // <<data_group_ranks[0]<<" "
  // <<data_group_ranks[1]
  <<" data_group_id: "<<data_group_id<<" "
  <<" in_group_id: "<<data_group_ranks[rank_in_owning_data_group]<<std::endl;
  // <<rank_in_owning_data_group<<std::endl;
  // // Initialize Data Parallel Group
  // const int32_t data_group_id = world_rank_with_offset % (horizontal_parallel_size);
  // const int32_t rank_in_owning_data_group = world_rank_with_offset / (horizontal_parallel_size);
  // std::vector<int32_t> data_group_ranks;
  // for (auto r = 0; r < data_parallel_size; r++) {
  //   data_group_ranks.push_back(data_group_id + horizontal_parallel_size * r);
  // }
  // groups_[WorkerGroupType::DataParallel] = {data_group_ranks, data_group_id,
  //                                           WorkerGroupType::DataParallel, rank_in_owning_data_group};

  // Horizontal Model Parallel Group
  const int32_t hori_group_id = world_rank / horizontal_parallel_size;
  const int32_t rank_in_owning_hori_group = world_rank % horizontal_parallel_size;
  std::vector<int32_t> hori_group_ranks;
  for (auto r = 0; r < horizontal_parallel_size; r++) {
    hori_group_ranks.push_back(hori_group_id * horizontal_parallel_size + r);
  }
    ORT_ENFORCE(
      hori_group_ranks[rank_in_owning_hori_group] == world_rank,
      "hori parallel distributed group cal wrong: ", world_rank, " ", hori_group_ranks[rank_in_owning_hori_group]);

  groups_[WorkerGroupType::HorizontalParallel] = {hori_group_ranks, hori_group_id,
                                                  WorkerGroupType::HorizontalParallel, rank_in_owning_hori_group};

  std::cout<<"** world rank["<<world_rank<<"] HP: horizontal_parallel_size: "<<horizontal_parallel_size<<" "
  // <<hori_group_ranks[0]<<" "
  // <<hori_group_ranks[1]
  <<" data_group_id: "<<hori_group_id
  <<" in_group_id: "<<hori_group_ranks[rank_in_owning_hori_group]<<std::endl;
  // <<" "<<rank_in_owning_hori_group<<std::endl;
  // // Horizontal Model Parallel Group
  // const int32_t hori_group_id = world_rank_with_offset / horizontal_parallel_size;
  // const int32_t rank_in_owning_hori_group = world_rank_with_offset % horizontal_parallel_size;
  // std::vector<int32_t> hori_group_ranks;
  // for (auto r = 0; r < horizontal_parallel_size; r++) {
  //   hori_group_ranks.push_back(hori_group_id * horizontal_parallel_size + r);
  // }
  // groups_[WorkerGroupType::HorizontalParallel] = {hori_group_ranks, hori_group_id,
  //                                                 WorkerGroupType::HorizontalParallel, rank_in_owning_hori_group};

  // Pipeline Model Parallel Group
  const int32_t pipeline_group_id = world_rank % (horizontal_parallel_size * data_parallel_size);
  const int32_t rank_in_owning_pipeline_group = world_rank / (horizontal_parallel_size * data_parallel_size);
  std::vector<int32_t> pipeline_group_ranks;
  for (auto r = 0; r < pipeline_stage_size; r++) {
    pipeline_group_ranks.push_back(pipeline_group_id + horizontal_parallel_size * data_parallel_size * r);
  }
      ORT_ENFORCE(
      pipeline_group_ranks[rank_in_owning_pipeline_group] == world_rank,
      "pipeline parallel distributed group cal wrong: ", world_rank, " ", pipeline_group_ranks[rank_in_owning_pipeline_group]);

  groups_[WorkerGroupType::ModelParallel] = {pipeline_group_ranks, pipeline_group_id,
                                                  WorkerGroupType::ModelParallel, rank_in_owning_pipeline_group};

  std::cout<<"** world rank["<<world_rank<<"] MP: pipeline_stage_size: "<<pipeline_stage_size<<" "
  <<" data_group_id: "<<pipeline_group_id
  <<" in_group_id: "<<pipeline_group_ranks[rank_in_owning_pipeline_group]<<std::endl;


  // const int32_t slice_index = world_rank / (horizontal_parallel_size * data_parallel_size);
  // const int32_t offset = (slice_index * horizontal_parallel_size * data_parallel_size);
  // auto world_rank_with_offset = world_rank - offset;

}

}  // namespace training
}  // namespace onnxruntime
