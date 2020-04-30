// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/models/runner/pipeline.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);
common::Status TransformGraphForPipeline(
    Graph& graph,
    std::string& forward_waited_event_name,
    std::string& forward_recorded_event_name,
    std::string& backward_waited_event_name,
    std::string& backward_recorded_event_name);

using CutInfo = std::vector<pipeline::PipelineContext::CutEdge>;
common::Status SplitGraphForPipeline(const Graph& graph, std::vector<CutInfo> cut_info, size_t pipeline_stage_id, const std::string& input_file_name,std::string& pipeline_partition_file_name);
}  // namespace training
}  // namespace onnxruntime
