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
    std::string& forward_waited_event_name,
    std::string& forward_recorded_event_name,
    std::string& backward_waited_event_name,
    std::string& backward_recorded_event_name,
    std::string& forward_wait_output_name,
    std::string& forward_record_output_name,
    std::string& backward_waited_output_name,
    std::string& backward_record_output_name,
    std::string& forward_waited_event_after_recv_name,
    std::string& forward_recorded_event_before_send_name,
    std::string& backward_waited_event_after_recv_name,
    std::string& backward_recorded_event_before_send_name);

Status ApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage);
}  // namespace training
}  // namespace onnxruntime
