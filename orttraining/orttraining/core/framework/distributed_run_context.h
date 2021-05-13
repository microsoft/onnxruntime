// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>

#include "core/common/common.h"

namespace onnxruntime {
namespace training {
enum WorkerGroupType {
  // The global view of all parallel workers.
  GlobalParallel = 0,
  // The view of all data parallel workers.
  DataParallel = 1,
  // The view of data parallel worker groups within a node.
  NodeLocalDataParallel = 2,
  // The view of data parallel worker groups aross nodes.
  CrossNodeDataParallel = 3,
  // The view of Megatron-style model parallel workers.
  HorizontalParallel = 4,
  // The view of pipeline model parallel workers
  PipelineParallel = 5,
  WorkerGroupTypeCount = 6,
};

struct WorkerGroup {
  std::vector<int32_t> ranks;  // array of global world rank
  int32_t group_id{-1};
  WorkerGroupType group_type;
  int32_t rank_in_group{-1};  // current worker' relative rank within this group, ranging from 0 to size-1

  std::string ToString() const {
    std::stringstream msg;
    msg << "group_type: " << group_type << ", group_id: " << group_id << ", rank in group:" << rank_in_group << ", world-rank:" << ranks.at(rank_in_group);
    msg << ", ranks: [";
    for (size_t i = 0; i < ranks.size(); ++i) {
      msg << ranks.at(i);
      if (i != ranks.size() - 1) {
        msg << ", ";
      }
    }
    msg << "]";

    return msg.str();
  }
};

struct DistributedRunConfig {
  int32_t world_rank{0};  // Get global world rank
  int32_t world_size{1};  // Get global world size
  int32_t local_rank{0};  // Get local rank on one physical node.
  int32_t local_size{1};  // Get local size of one physical node.
  int32_t data_parallel_size{1};
  int32_t horizontal_parallel_size{1};
  int32_t pipeline_stage_size{1};
};

// This function returns the corresponding pipeline stage id for the given world rank.
inline int32_t GetPipelineStageId(const int32_t world_rank,
                                  const int32_t horizontal_parallel_size,
                                  const int32_t data_parallel_size) {
  return world_rank / (data_parallel_size * horizontal_parallel_size);
}

// Context managing global distribute run config, also responsible for splitting workers into groups
// using passed-in's parallel sizes.
// For example, workers [0, 1, 2, ..., 63], using 4-way horizontal model parallel + 16-way data parallel,
// will be splitted into:
//     4 data parallel groups: [0, 4, ..., 60], [1, 5, ..., 61], ..., [3, 7, ..., 63]
//     16 horizontal parallel groups: [0, 1, 2, 3], [4, 5, 6, 7], ..., [60, 61, 62, 63]
class DistributedRunContext {
 public:
  static DistributedRunContext& CreateInstance(DistributedRunConfig config) {
    return DistributedRunContext::GetOrCreateInstance(config.world_rank, config.world_size,
                                                      config.local_rank, config.local_size,
                                                      config.data_parallel_size,
                                                      config.horizontal_parallel_size,
                                                      config.pipeline_stage_size);
  }

#ifndef SHARED_PROVIDER
  static DistributedRunContext& GetInstance() {
    return DistributedRunContext::GetOrCreateInstance();
  }
#else
  DistributedRunContext& GetInstance() { return Provider_GetHost()->GetDistributedRunContextInstance(); }
#endif
  /* SHORTCUT FUNCTIONS START */

  static DistributedRunConfig& RunConfig() {
    return DistributedRunContext::GetInstance().GetRunConfig();
  }

  // Utility function to return string representation of each WorkerGroupType
  static std::string GetWorkerGroupName(WorkerGroupType group) {
    switch (group) {
      case WorkerGroupType::GlobalParallel:
        return "GlobalParallel";
      case WorkerGroupType::DataParallel:
        return "DataParallel";
      case WorkerGroupType::NodeLocalDataParallel:
        return "NodeLocalDataParallel";
      case WorkerGroupType::CrossNodeDataParallel:
        return "CrossNodeDataParallel";
      case WorkerGroupType::HorizontalParallel:
        return "HorizontalParallel";
      case WorkerGroupType::PipelineParallel:
        return "PipelineParallel";
      default:
        ORT_THROW("Unsupported distributed worker group type.");
    }
  }

  // Get current worker' rank within the specified group,
  // value ranges from 0 ~ group_size -1
  static int32_t RankInGroup(WorkerGroupType group_type) {
    return DistributedRunContext::GetInstance().GetWorkerGroup(group_type).rank_in_group;
  }

  static int32_t GroupId(WorkerGroupType group_type) {
    return DistributedRunContext::GetInstance().GetWorkerGroup(group_type).group_id;
  }

  static std::vector<int32_t> GetRanks(WorkerGroupType group_type) {
    return DistributedRunContext::GetInstance().GetWorkerGroup(group_type).ranks;
  }

  // Get total rank of specified group.
  static int32_t GroupSize(WorkerGroupType group_type) {
    return static_cast<int32_t>(DistributedRunContext::GetInstance().GetWorkerGroup(group_type).ranks.size());
  }

  /* SHORTCUT FUNCTIONS END */

  DistributedRunConfig& GetRunConfig() {
    return params_;
  }

  // Get specified worker group.
  WorkerGroup& GetWorkerGroup(WorkerGroupType group_type) {
    assert(group_type < WorkerGroupTypeCount);
    return groups_[group_type];
  }

 protected:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DistributedRunContext);

  DistributedRunContext(int32_t world_rank, int32_t world_size, int32_t local_rank, int32_t local_size,
                        int32_t data_parallel_size, int32_t horizontal_parallel_size, int32_t pipeline_stage_size = 1);

  static DistributedRunContext& GetOrCreateInstance(int32_t world_rank = 0, int32_t world_size = 1,
                                                    int32_t local_rank = 0, int32_t local_size = 1,
                                                    int32_t data_parallel_size = 1,
                                                    int32_t horizontal_parallel_size = 1,
                                                    int32_t pipeline_stage_size = 1) {
    static DistributedRunContext instance(world_rank, world_size, local_rank, local_size,
                                          data_parallel_size, horizontal_parallel_size, pipeline_stage_size);
    return instance;
  }

  DistributedRunConfig params_;

  // Groups containing current worker
  std::vector<WorkerGroup> groups_;
};

}  // namespace training
}  // namespace onnxruntime
