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

Status ApplyPipelinePartitionToMainGraph(Graph& graph,
    std::map<Node*, int>& op_to_rank,
    bool is_training,
    int pipeline_stage_id,
    int nstages);

}  // namespace training
}  // namespace onnxruntime
