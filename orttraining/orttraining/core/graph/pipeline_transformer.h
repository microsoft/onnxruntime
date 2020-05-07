// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/models/runner/pipeline.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);
common::Status TransformGraphForPipeline(
    Graph& graph,
    std::string& forward_waited_event_name,
    std::string& forward_recorded_event_name,
    std::string& backward_waited_event_name,
    std::string& backward_recorded_event_name);

// using CutInfo = std::vector<CutEdge>;
common::Status SplitGraphForPipeline(Graph& graph,
                                     std::vector<TrainingSession::TrainingConfiguration::CutInfo> cut_info,
                                     size_t pipeline_stage_id,
                                     size_t num_pipeline_stages,
                                     std::string& pipeline_partition_file_name);

class PipelineTransformer : public GraphTransformer {
 public:
  PipelineTransformer(std::vector<TrainingSession::TrainingConfiguration::CutInfo> cut_info,
                      size_t pipeline_stage_id,
                      size_t num_pipeline_stages,
                      const std::string& pipeline_partition_file_name,
                      const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("PipelineTransformer", compatible_execution_providers),
        split_edge_groups_(cut_info),
        pipeline_stage_id_(pipeline_stage_id),
        num_pipeline_stages_(num_pipeline_stages),
        pipeline_partition_file_name_(pipeline_partition_file_name) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

 private:
 common::Status SplitGraph(Graph& graph, std::vector<Node*>& send_nodes, std::vector<Node*>& recv_nodes) const;
 common::Status GenerateSubgraph(Graph& graph, const Node* start_node) const;
  std::vector<TrainingSession::TrainingConfiguration::CutInfo> split_edge_groups_;
  size_t pipeline_stage_id_;
  size_t num_pipeline_stages_;
  std::string pipeline_partition_file_name_;
};
}  // namespace training
}  // namespace onnxruntime
