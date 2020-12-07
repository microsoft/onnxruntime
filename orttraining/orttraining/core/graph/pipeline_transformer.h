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

// TODO(jufranc): when adapting this code to partition training graphs, add
// a boolean is_training as parameter.
Status ApplyPipelinePartitionToMainGraph(Graph& graph,
    std::map<Node*, int>& op_to_stage,
    int pipeline_stage_id,
    int num_stages);

// First of two functions to obtain a mapping between operators and stage ids.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - id_to_stage maps string identifiers of operators and stage ids. Each
// operator is identified with the name of any of its outputs.
//   - op_to_stage maps pointers to operators and stage ids. This is the output
// of this function.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(Graph& graph,
    const std::map<std::string, int>& id_to_stage,
    std::map<Node*, int>& op_to_stage,
    int num_stages);

// Second of two functions to obtain a mapping between operators and stage ids.
// This version in particular converts a list of graph cuts (i.e., CutInfo)
// into a mapping between operators and stages.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - cuts describes all the cut points as defined by the user (c.f., CutInfo
// type definition.
//   - op_to_stage maps pointers to operators and stage ids. This is the output
// of this function.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cuts,
    std::map<Node*, int>& op_to_stage,
    int num_stages);

}  // namespace training
}  // namespace onnxruntime
