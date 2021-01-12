// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);

Status TransformGraphForPipeline(
    Graph& graph,
    const std::unordered_set<std::string>& initializer_names_to_preserve,
    pipeline::PipelineTensorNames& pipeline_tensor_names);

// Partitions the graph into num_stages subgraphs, as defined in op_to_stage map,
// which maps operators to stage ids. After the graph is partitioned, it drops
// all the tensors and operators that do not belong to the subgraph pipeline_stage_id.
// TODO(jufranc): when adapting this code to partition training graphs, add
// a boolean is_training as parameter.
Status ApplyPipelinePartitionToMainGraph(Graph& graph,
    const std::map<const Node*, int>& op_to_stage,
    const int pipeline_stage_id,
    const int num_stages,
    const std::vector<int32_t>& rank_ids);

// First of two functions to obtain a mapping between operators and stage ids.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - id_to_stage maps string identifiers of operators and stage ids. Each
// operator is identified with the name of any of its outputs.
//   - op_to_stage keeps the output of this function, where op_to_stage[node_ptr]
// is the pipeline stage ID of the pointed node.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(const Graph& graph,
    const std::map<std::string, int>& id_to_stage,
    std::map<const Node*, int>& op_to_stage,
    const int num_stages);

// Second of two functions to obtain a mapping between operators and stage ids.
// This version in particular converts a list of graph cuts (i.e., CutInfo)
// into a mapping between operators and stages.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - cuts describes all the cut points as defined by the user (c.f., CutInfo
// type definition.
//   - op_to_stage keeps the output of this function, where op_to_stage[node_ptr]
// is the pipeline stage ID of the pointed node.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(const Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cuts,
    std::map<const Node*, int>& op_to_stage,
    const int num_stages);

}  // namespace training
}  // namespace onnxruntime
