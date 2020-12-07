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

Status GetDeviceAssignmentMap(Graph& graph,
                              const std::map<std::string, int>& id_to_stage,
                              std::map<Node*, int>& op_to_stage,
                              int num_stages);

Status GetDeviceAssignmentMap(Graph& graph,
                              const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cuts,
                              std::map<Node*, int>& op_to_stage,
                              int num_stages);

}  // namespace training
}  // namespace onnxruntime
